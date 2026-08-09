"""
src/rag/parser.py
=================
Estrategias de procesamiento de texto y fragmentación jerárquica (Parent-Child) con LlamaIndex.

Responsabilidades:
  - Aplicar fragmentación jerárquica (HierarchicalNodeParser) a nodos de texto o documentos
  - Separar nodos finales (padres e hijos) de los nodos hoja (leaf nodes) para la indexación RAG
"""

from __future__ import annotations

from typing import Sequence
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.schema import BaseNode


def build_hierarchical_nodes(
    nodes: Sequence[BaseNode],
    chunk_sizes: list[int] | None = None,
    chunk_overlap: int = 30,
) -> tuple[list[BaseNode], list[BaseNode]]:
    """
    Toma una lista de nodos de dominio y aplica fragmentación jerárquica (Parent-Child),
    generando relaciones entre bloques de mayor tamaño (padres) y menor tamaño (hijos).

    Args:
        nodes: Lista de nodos de entrada (p. ej. nodos de dominio de Redmine).
        chunk_sizes: Tamaños de bloque para los niveles jerárquicos (por defecto [1024, 256]).
        chunk_overlap: Solapamiento entre fragmentos (por defecto 30 tokens).

    Returns:
        Una tupla con (final_nodes, leaf_nodes):
          - final_nodes: Todos los nodos de la jerarquía (padres + hijos) para guardar en docstore.
          - leaf_nodes: Nodos hoja únicamente, para ser indexados vectorialmente en Qdrant.
    """
    if chunk_sizes is None:
        chunk_sizes = [1024, 256]

    hierarchical_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=chunk_sizes,
        chunk_overlap=chunk_overlap,
    )

    final_nodes = hierarchical_parser.get_nodes_from_documents(nodes)
    leaf_nodes = get_leaf_nodes(final_nodes)

    return final_nodes, leaf_nodes
