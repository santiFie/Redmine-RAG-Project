"""
src/rag/engine.py
=================
Motor RAG basado en LlamaIndex con Qdrant como Vector Store.

Responsabilidades:
  - Inicializar el LLM y el modelo de embeddings
  - Conectar con Qdrant como vector store persistente y SimpleDocumentStore para la jerarquía de nodos
  - Orquestar la ingesta en 3 Fases: Domain Parsing -> Text Parsing Jerárquico -> Indexación
  - Exponer un QueryEngine para responder preguntas en lenguaje natural
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any

from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.retrievers import (
    AutoMergingRetriever,
    BaseRetriever,
    VectorIndexAutoRetriever,
)
from llama_index.core.schema import (
    NodeRelationship,
    NodeWithScore,
    QueryBundle,
    RelatedNodeInfo,
    TextNode,
)
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding
from llama_index.storage.docstore.postgres import PostgresDocumentStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, QdrantClient

from src.rag.schemas import REDMINE_VECTOR_STORE_INFO
from src.rag.utils.get_llm import _LLM_REGISTRY
from src.rag.utils.parser import build_hierarchical_nodes
from src.redmine.parser import parse_redmine_issue_to_nodes

load_dotenv()

logger = logging.getLogger(__name__)


# ==============================================================================
# Retriever async-compatible
# ==============================================================================

class AsyncAutoMergingRetriever(AutoMergingRetriever):
    """
    Subclase de AutoMergingRetriever que sobreescribe ``_aretrieve`` para hacer
    el paso de búsqueda vectorial de forma verdaderamente asíncrona.

    La clase base de LlamaIndex tiene ``_aretrieve`` como fallback al método
    síncrono. Aquí reemplazamos sólo la llamada al vector retriever por su
    variante async (``aretrieve``), que usa ``AsyncQdrantClient`` internamente
    para no bloquear el event loop. El merging posterior (``_try_merging``) opera
    únicamente sobre estructuras en memoria y el docstore, por lo que su costo
    de I/O es despreciable y se mantiene síncrono.
    """

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        # Paso costoso: embedding + búsqueda en Qdrant → async nativo
        initial_nodes = await self._vector_retriever.aretrieve(query_bundle)

        # Merging jerárquico: opera en memoria/docstore → síncrono aceptable
        cur_nodes, is_changed = self._try_merging(initial_nodes)
        while is_changed:
            cur_nodes, is_changed = self._try_merging(cur_nodes)

        cur_nodes.sort(key=lambda x: x.get_score(), reverse=True)
        logger.debug(
            "Hybrid retrieval: %d nodos iniciales → %d tras auto-merge",
            len(initial_nodes),
            len(cur_nodes),
        )
        return cur_nodes

class RAGEngine:
    """
    Motor de Retrieval-Augmented Generation con LlamaIndex, Qdrant y estrategia Parent-Child.
    """

    def __init__(
        self,
        collection_name: str | None = None,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
        llm_provider: str | None = None,
    ) -> None:

        # Inicilizar modelos
        self.llm_provider = llm_provider or os.getenv("LLM_PROVIDER", "groq").lower()
        factory = _LLM_REGISTRY.get(self.llm_provider)
        
        if factory is None:
            raise ValueError(f"LLM Provider no soportado: {self.llm_provider}")
        self._llm = factory()

        self._embed_model = HuggingFaceInferenceAPIEmbedding(
            model_name=os.getenv("EMBED_MODEL", "BAAI/bge-m3"),
            token=os.getenv("HUGGINGFACE_API_KEY", ""),
        )

        Settings.llm = self._llm
        Settings.embed_model = self._embed_model

        # Inicializar Qdrant
        self.collection_name = collection_name or "redmine_docs"
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        
        raw_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        self.qdrant_api_key = raw_key.strip() if raw_key and raw_key.strip() else None

        logger.info(
            "RAGEngine init: provider=%s embed=%s coleccion=%s qdrant=%s",
            self.llm_provider,
            os.getenv("EMBED_MODEL", "BAAI/bge-m3"),
            self.collection_name,
            self.qdrant_url,
        )

        self._qdrant = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
        self._async_qdrant = AsyncQdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)

        # Docstore + VectorStore para almacenar padres y vectorizar hijos
        self._vector_store = QdrantVectorStore(
            client=self._qdrant,
            aclient=self._async_qdrant,
            collection_name=self.collection_name,
            enable_hybrid=True,
            fastembed_sparse_model="Qdrant/bm25"
        )
        self._storage_context = self._init_storage_context()
        self._index = self._load_or_create_index()

        # Warmup del provider mapping de HuggingFace.
        # `_fetch_inference_provider_mapping` está decorada con @lru_cache: la primera
        # llamada hace un HTTP síncrono a la Hub API para resolver el proveedor del modelo.
        # Disparándola aquí (en tiempo de importación, antes del event loop de LangGraph)
        # se pre-popula el caché. Las llamadas async posteriores desde el event loop
        # encontrarán el caché caliente y no harán I/O bloqueante.
        with contextlib.suppress(Exception):
            self._embed_model.get_query_embedding("warmup")


    def parse_redmine_issue_to_nodes(self, issue_data: dict[str, Any]) -> list[TextNode]:
        """
        Delegación al parser del dominio Redmine (mantenido por compatibilidad).
        """
        return parse_redmine_issue_to_nodes(issue_data)


    def index_documents(self, documents: list[Any]) -> None:
        """
        Indexa una lista de objetos Document / Node de LlamaIndex en Qdrant.
        """
        if not documents:
            return
        
        self._index.insert_documents(documents)
        

    def index_redmine_issues(self, issues: list[dict[str, Any]]) -> None:
        """
        Ejecuta el flujo de indexación completo.
        """
        for issue in issues:
            issue_id = str(issue["id"])

            # ELIMINAR versión vieja
            try:
                self._index.delete_ref_doc(issue_id, delete_from_docstore=True)
            except Exception:
                logger.debug("Issue %s no existía previamente, se omite el borrado", issue_id)

            # FASE 1: Domain Parsing
            domain_nodes = self.parse_redmine_issue_to_nodes(issue)

            # Vincular con ref_doc_id
            for node in domain_nodes:
                node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=str(issue_id))

            # FASE 2: Text Parsing (Reglas de LlamaIndex para Parent-Child)
            final_nodes, leaf_nodes = build_hierarchical_nodes(domain_nodes)

            logger.info(
                "Indexando issue #%s: %d nodos dominio → %d nodo(s) hoja",
                issue_id,
                len(domain_nodes),
                len(leaf_nodes),
            )

            # FASE 3: Indexación y Almacenamiento
            # Guardar la estructura del árbol jerárquico completo en Postgres
            self._storage_context.docstore.add_documents(final_nodes)

            # Guardar e indexar únicamente los nodos hoja en Qdrant
            self._index.insert_nodes(leaf_nodes)


    def query(self, question: str, top_k: int = 5) -> str:
        """Método síncrono (mantenido para compatibilidad y tests)."""
        retriever = self._build_base_retriever(top_k)
        # Sync Merging Retriever
        merging_retriever = AutoMergingRetriever(
            vector_retriever=retriever,
            storage_context=self._storage_context,
            simple_ratio_thresh=0.5,
            verbose=False,
        )
        nodes = merging_retriever.retrieve(question)
        logger.info(
            "Query sync '%s...' → %d nodos | scores=%s",
            question[:50],
            len(nodes),
            [round(n.score or 0.0, 3) for n in nodes],
        )
        return self._format_context(nodes)

    async def aquery(
        self, question: str, top_k: int = 5
    ) -> str:
        """
        Variante async de ``query``: usa ``AsyncAutoMergingRetriever`` para que
        la búsqueda vectorial en Qdrant no bloquee el event loop de LangGraph.
        """
        retriever = self._build_base_retriever(top_k)
        # Async Merging Retriever
        merging_retriever = AsyncAutoMergingRetriever(
            vector_retriever=retriever,
            storage_context=self._storage_context,
            simple_ratio_thresh=0.5,
            verbose=False,
        )
        nodes = await merging_retriever.aretrieve(question)
        logger.info(
            "Query async '%s...' → %d nodos | scores=%s",
            question[:50],
            len(nodes),
            [round(n.score or 0.0, 3) for n in nodes],
        )
        return self._format_context(nodes)

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _build_base_retriever(self, top_k: int = 5) -> BaseRetriever:
        logger.info(
            "Construyendo base retriever: modo=HYBRID alpha=0.5 sparse_top_k=5 top_k=%d",
            top_k,
        )
        base_auto_retriever = VectorIndexAutoRetriever(
            index=self._index,
            vector_store_info=REDMINE_VECTOR_STORE_INFO,
            vector_store_query_mode=VectorStoreQueryMode.HYBRID,
            similarity_top_k=top_k,
            empty_query_top_k=10,
            extra_retriever_kwargs={
                "alpha": 0.5,       # 0.5 balancea denso (embeddings) y sparse (BM25)
                "sparse_top_k": 5,  # Top-k candidatos para la rama léxica
            },
        )

        return base_auto_retriever


    def _format_context(self, nodes: list[NodeWithScore]) -> str:
        """Convierte una lista de nodos recuperados en texto de contexto."""
        full_context = ""
        for node_with_score in nodes:
            node = node_with_score.node
            full_context += f"Score: {node_with_score.score:.4f}\n"
            full_context += f"Texto: {node.get_content()}\n"
            full_context += f"Metadatos: {node.metadata}\n\n"
        return full_context


    def _init_storage_context(self) -> StorageContext:
        self._docstore = PostgresDocumentStore.from_uri(
            uri=os.getenv("POSTGRES_URI", ""),
            table_name="docstore"
        )
        
        return StorageContext.from_defaults(
            vector_store=self._vector_store,
            docstore=self._docstore,
        )
        

    def _load_or_create_index(self) -> Any:
        """
        Carga el índice desde Qdrant si ya existe, o crea uno nuevo.
        """
        if self._qdrant.collection_exists(self.collection_name):
            # Restaurar docstore (jerarquía Parent-Child) desde Postgres
            self._storage_context = StorageContext.from_defaults(
                vector_store=self._vector_store,
                docstore=self._docstore,
            )
            return VectorStoreIndex.from_vector_store(
                vector_store=self._vector_store,
                storage_context=self._storage_context,
            )
        else:
            return VectorStoreIndex(
                nodes=[],
                storage_context=self._storage_context,
        )
