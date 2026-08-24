"""
Esquemas de metadatos e información del Vector Store para LlamaIndex.

Responsabilidades:
    - Definir la estructura de MetadataInfo correspondiente a los metadatos indexados en Qdrant.
    - Exponer VectorStoreInfo para su uso en VectorIndexAutoRetriever u otros retrievers guiados por LLM.
"""

from __future__ import annotations

from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo

# Definición de los metadatos presentes en los nodos de Redmine
REDMINE_METADATA_INFO: list[MetadataInfo] = [
    MetadataInfo(
        name="issue_id",
        type="int",
        description="Identificador numérico único del ticket o issue en Redmine.",
    ),
    MetadataInfo(
        name="entity_type",
        type="str",
        description="Tipo de entidad del nodo: 'issue_description' para la descripción principal del ticket o 'issue_comment' para un comentario/journal.",
    ),
    MetadataInfo(
        name="project",
        type="str",
        description="Nombre del proyecto en Redmine al que pertenece el ticket.",
    ),
    MetadataInfo(
        name="status",
        type="str",
        description="Estado del issue (por ejemplo: 'Nueva', 'En curso', 'Cerrada', etc.). Presente en la descripción principal.",
    ),
    MetadataInfo(
        name="author",
        type="str",
        description="Nombre del autor o creador del issue. Presente en la descripción principal.",
    ),
    MetadataInfo(
        name="comment_author",
        type="str",
        description="Nombre del usuario que realizó el comentario. Presente en nodos de tipo 'issue_comment'.",
    ),
    MetadataInfo(
        name="created_on",
        type="str",
        description="Fecha y hora de creación del comentario en formato ISO 8601 (p. ej. '2024-01-15T10:00:00Z'). Presente en comentarios.",
    ),
]

# Información descriptiva del contenido del Vector Store para el auto-retriever
REDMINE_VECTOR_STORE_INFO = VectorStoreInfo(
    content_info="Tickets, descripciones principales, incidencias y comentarios/historial de issues de Redmine.",
    metadata_info=REDMINE_METADATA_INFO,
)
