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

import os
from typing import Any

from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import MetadataFilters
from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding
from llama_index.storage.docstore.postgres import PostgresDocumentStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, QdrantClient

from src.rag.utils.get_llm import _LLM_REGISTRY
from src.rag.utils.parser import build_hierarchical_nodes
from src.redmine.parser import parse_redmine_issue_to_nodes

load_dotenv()

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

        self._qdrant = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
        self._async_qdrant = AsyncQdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)

        # Docstore + VectorStore para almacenar padres y vectorizar hijos
        self._vector_store = QdrantVectorStore(
            client=self._qdrant,
            aclient=self._async_qdrant,
            collection_name=self.collection_name,
        )
        self._storage_context = self._init_storage_context()
        self._index = self._load_or_create_index()


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
                pass # Si no existía previamente, ignorar

            # FASE 1: Domain Parsing
            domain_nodes = self.parse_redmine_issue_to_nodes(issue)

            # Vincular con ref_doc_id
            for node in domain_nodes:
                node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=str(issue_id))

            # FASE 2: Text Parsing (Reglas de LlamaIndex para Parent-Child)
            final_nodes, leaf_nodes = build_hierarchical_nodes(domain_nodes)

            # FASE 3: Indexación y Almacenamiento
            # Guardar la estructura del árbol jerárquico completo en Postgres
            self._storage_context.docstore.add_documents(final_nodes)

            # Guardar e indexar únicamente los nodos hoja en Qdrant
            self._index.insert_nodes(leaf_nodes)


    def query(self, question: str, top_k: int = 5, filter: MetadataFilters | None = None) -> str:
        """Método síncrono para ejecutar dentro de un thread aislado."""
        retriever = AutoMergingRetriever(
            vector_retriever=self._index.as_retriever(similarity_top_k=top_k, filters=filter),
            storage_context=self._storage_context,
            simple_ratio_thresh=0.1
        )

        # LlamaIndex ejecuta su flujo síncrono sin colisionar con event loops
        nodes = retriever.retrieve(question)

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
