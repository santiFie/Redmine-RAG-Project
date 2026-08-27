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

# Prompt template personalizado en español para VectorIndexAutoRetriever
REDMINE_AUTO_RETRIEVER_PROMPT_TMPL = """Tu objetivo es estructurar la consulta del usuario para que coincida con el esquema de solicitud estructurada proporcionado a continuación.

<< Structured Request Schema >>
Al responder, utiliza un bloque de código markdown con un objeto JSON con el siguiente esquema:

{schema_str}

La cadena `query` debe contener ÚNICAMENTE el texto relevante para la búsqueda semántica en el contenido de los documentos. Cualquier condición o filtro (proyecto, autor, estado, ID) DEBE ser extraído a la sección de filtros y ELIMINADO de la cadena `query`.

Reglas:
1. Asegúrate de que los filtros hagan referencia exclusivamente a los atributos definidos en la fuente de datos.
2. Si no hay filtros aplicables, devuelve [] en filters.
3. Si el usuario pide explícitamente una cantidad de documentos (p. ej., "los últimos 3 comentarios"), establece top_k con ese número; de lo contrario, déjalo en null.
4. Mantén la cadena `query` en el mismo idioma de la consulta (español).

<< Example 1. >>
Data Source:
```json
{{
    "metadata_info": [
        {{
            "name": "project",
            "type": "str",
            "description": "Nombre del proyecto en Redmine al que pertenece el ticket."
        }},
        {{
            "name": "status",
            "type": "str",
            "description": "Estado del issue ('Nueva', 'En curso', 'Cerrada', etc.)."
        }},
        {{
            "name": "author",
            "type": "str",
            "description": "Nombre del autor o creador del issue."
        }}
    ],
    "content_info": "Tickets e incidencias de Redmine"
}}
```

User Query:
¿Qué incidencias abiertas reportó Juan Pérez sobre lentitud en la base de datos en el proyecto Core?

Structured Request:
```json
{{"query":"lentitud en la base de datos","filters":[{{"key":"project","value":"Core","operator":"=="}},{{"key":"author","value":"Juan Pérez","operator":"=="}},{{"key":"status","value":"Cerrada","operator":"!="}}],"top_k":null}}
```

<< Example 2. >>
Data Source:
```json
{{
    "metadata_info": [
        {{
            "name": "issue_id",
            "type": "int",
            "description": "Identificador numérico único del ticket o issue en Redmine."
        }},
        {{
            "name": "entity_type",
            "type": "str",
            "description": "Tipo de entidad del nodo: 'issue_description' o 'issue_comment'."
        }},
        {{
            "name": "comment_author",
            "type": "str",
            "description": "Nombre del usuario que realizó el comentario."
        }}
    ],
    "content_info": "Historial y comentarios de tickets en Redmine"
}}
```

User Query:
Muéstrame los comentarios de María en el ticket 1042 sobre el despliegue

Structured Request:
```json
{{"query":"despliegue","filters":[{{"key":"issue_id","value":1042,"operator":"=="}},{{"key":"entity_type","value":"issue_comment","operator":"=="}},{{"key":"comment_author","value":"María","operator":"=="}}],"top_k":null}}
```

<< Example 3. >>
Data Source:
```json
{info_str}
```

User Query:
{query_str}

Structured Request:
"""
