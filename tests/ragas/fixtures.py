"""
tests/ragas/fixtures.py
========================
Dataset de evaluación para el RAGEngine con Ragas.

Contiene:
  - REDMINE_ISSUES_FIXTURE  : issues de Redmine sintéticos que se indexan antes de evaluar.
  - RAGAS_TESTSET           : lista de dicts con las claves que Ragas necesita:
      · user_input          : pregunta del usuario
      · reference           : respuesta esperada (ground truth)
      · reference_contexts  : fragmentos del corpus que justifican la respuesta

Las preguntas cubren los cuatro tipos de métricas que se evaluarán:
  - Faithfulness            : la respuesta no inventa datos ajenos al contexto
  - AnswerRelevancy         : la respuesta es pertinente a la pregunta
  - ContextPrecision        : los contextos recuperados son relevantes para la pregunta
  - ContextRecall           : la respuesta de referencia puede derivarse del contexto recuperado
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Issues sintéticos de Redmine — se indexan en el RAGEngine durante el test
# ---------------------------------------------------------------------------

REDMINE_ISSUES_FIXTURE: list[dict] = [
    {
        "id": 101,
        "subject": "Error 500 en el endpoint /api/projects al filtrar por estado",
        "description": (
            "Al realizar una petición GET a /api/projects?status=active se obtiene un "
            "HTTP 500. El error ocurre únicamente cuando el parámetro 'status' se "
            "combina con el filtro 'assigned_to_id'. El stack trace indica un "
            "NullPointerException en la capa de repositorio (ProjectRepository.findByStatusAndAssignee). "
            "Prioridad: Alta. Entorno: producción."
        ),
        "status": {"name": "En progreso"},
        "author": {"name": "Ana García"},
        "project": {"name": "API Backend"},
        "journals": [
            {
                "id": 201,
                "notes": (
                    "Reproduje el error localmente. La causa raíz es que "
                    "'assigned_to_id' puede ser NULL en la base de datos cuando un "
                    "proyecto no tiene asignado un responsable. El query ORM no maneja "
                    "ese caso. Propongo añadir un IS NOT NULL en el WHERE o cambiar a "
                    "una LEFT JOIN."
                ),
                "user": {"name": "Carlos López"},
                "created_on": "2024-01-15T10:23:00Z",
            },
            {
                "id": 202,
                "notes": (
                    "Fix implementado en rama fix/project-null-assignee. "
                    "Se añadió la condición IS NOT NULL y se agregaron tests de "
                    "integración para el caso edge. Listo para revisión."
                ),
                "user": {"name": "Carlos López"},
                "created_on": "2024-01-16T09:45:00Z",
            },
        ],
    },
    {
        "id": 102,
        "subject": "Implementar autenticación OAuth2 para la integración con GitHub",
        "description": (
            "Se requiere implementar el flujo OAuth2 para permitir que los usuarios "
            "inicien sesión con su cuenta de GitHub. El sistema debe solicitar los "
            "scopes 'read:user' y 'user:email'. Una vez obtenido el token, se debe "
            "crear o actualizar el perfil del usuario en nuestra base de datos "
            "usando el email como identificador único. Tecnologías: Python 3.12, "
            "FastAPI, librería authlib."
        ),
        "status": {"name": "Nuevo"},
        "author": {"name": "María Torres"},
        "project": {"name": "Auth Service"},
        "journals": [
            {
                "id": 203,
                "notes": (
                    "Revisar si authlib soporta correctamente el flujo de "
                    "Authorization Code con PKCE, ya que GitHub lo recomienda para "
                    "aplicaciones web. Documentación de referencia: "
                    "https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/authorizing-oauth-apps"
                ),
                "user": {"name": "Luis Pérez"},
                "created_on": "2024-01-20T14:00:00Z",
            }
        ],
    },
    {
        "id": 103,
        "subject": "Degradación de rendimiento en consultas al vector store con más de 10k documentos",
        "description": (
            "A partir de los 10.000 documentos indexados en Qdrant, las consultas de "
            "similitud superan los 2 segundos de latencia. En entornos de producción "
            "con 50k+ documentos el tiempo sube a 8-10 segundos. Se sospecha que la "
            "colección no tiene configurado un índice HNSW apropiado. "
            "El parámetro m (número de conexiones por nodo) está en el valor por defecto (16). "
            "Aumentarlo a 32 o 64 podría mejorar la recall y reducir la latencia."
        ),
        "status": {"name": "En análisis"},
        "author": {"name": "Diego Fernández"},
        "project": {"name": "RAG Infrastructure"},
        "journals": [
            {
                "id": 204,
                "notes": (
                    "Realizamos pruebas de carga con m=32 y ef_construct=200. "
                    "La latencia de búsqueda bajó de 8s a 1.2s con 50k documentos. "
                    "La recall a top-10 mejoró de 0.87 a 0.94. "
                    "Recomiendo aplicar estos parámetros HNSW en la colección de producción."
                ),
                "user": {"name": "Sofía Ruiz"},
                "created_on": "2024-01-22T16:30:00Z",
            }
        ],
    },
    {
        "id": 104,
        "subject": "Agregar soporte para exportar reportes en formato CSV y Excel",
        "description": (
            "Los usuarios solicitan poder exportar el listado de issues filtrado "
            "en formato CSV y XLSX. El CSV debe usar UTF-8 con BOM para compatibilidad "
            "con Excel en Windows. El XLSX debe incluir la fila de encabezados en negrita "
            "y aplicar autofit a las columnas. Columnas requeridas: ID, Título, Estado, "
            "Autor, Proyecto, Fecha de creación, Fecha de actualización."
        ),
        "status": {"name": "En progreso"},
        "author": {"name": "Elena Martínez"},
        "project": {"name": "Reporting Module"},
        "journals": [
            {
                "id": 205,
                "notes": (
                    "Usando la librería openpyxl para el XLSX y el módulo csv estándar "
                    "de Python para el CSV. El BOM se añade escribiendo b'\\xef\\xbb\\xbf' "
                    "al inicio del archivo. Los tests unitarios están en tests/unit/test_export.py."
                ),
                "user": {"name": "Elena Martínez"},
                "created_on": "2024-01-25T11:00:00Z",
            }
        ],
    },
    {
        "id": 105,
        "subject": "Pipeline CI/CD: fallos intermitentes en la etapa de tests de integración",
        "description": (
            "En GitHub Actions, la etapa 'integration-tests' falla aproximadamente "
            "1 de cada 5 ejecuciones con el error: 'Connection refused: localhost:5432'. "
            "El servicio PostgreSQL del workflow parece no estar listo cuando los tests "
            "comienzan. Se usa 'services: postgres' en el YAML del workflow. "
            "La condición 'health: starting' no es suficiente para garantizar que "
            "la BD esté aceptando conexiones."
        ),
        "status": {"name": "Resuelto"},
        "author": {"name": "Roberto Silva"},
        "project": {"name": "DevOps"},
        "journals": [
            {
                "id": 206,
                "notes": (
                    "Solución: añadir una etapa 'wait-for-postgres' que ejecute "
                    "pg_isready en un loop con reintentos antes de correr los tests. "
                    "También se cambió la condición health check a "
                    "'options: --health-cmd pg_isready --health-interval 10s --health-retries 5'. "
                    "Con esto se eliminaron los fallos intermitentes."
                ),
                "user": {"name": "Roberto Silva"},
                "created_on": "2024-01-28T08:15:00Z",
            }
        ],
    },
]


# ---------------------------------------------------------------------------
# Testset para Ragas
# Cada entrada tiene:
#   user_input        — pregunta a hacerle al RAG
#   reference         — respuesta correcta de referencia (ground truth)
#   reference_contexts — fragmentos textuales del corpus que respaldan la respuesta
# ---------------------------------------------------------------------------

RAGAS_TESTSET: list[dict[str, str | list[str]]] = [
    # ── Issue 101 ── Error 500 en /api/projects ──────────────────────────────
    {
        "user_input": "¿Cuál es la causa raíz del error 500 en el endpoint /api/projects?",
        "reference": (
            "La causa raíz del error HTTP 500 en el endpoint /api/projects es un "
            "NullPointerException en ProjectRepository.findByStatusAndAssignee. "
            "Esto ocurre porque el campo 'assigned_to_id' puede ser NULL en la base "
            "de datos cuando un proyecto no tiene responsable asignado, y el query "
            "ORM no contempla ese caso."
        ),
        "reference_contexts": [
            (
                "Issue #101: Error 500 en el endpoint /api/projects al filtrar por estado\n\n"
                "Al realizar una petición GET a /api/projects?status=active se obtiene un "
                "HTTP 500. El error ocurre únicamente cuando el parámetro 'status' se "
                "combina con el filtro 'assigned_to_id'. El stack trace indica un "
                "NullPointerException en la capa de repositorio (ProjectRepository.findByStatusAndAssignee)."
            ),
            (
                "Comentario en Issue #101 por Carlos López:\n"
                "Reproduje el error localmente. La causa raíz es que 'assigned_to_id' puede "
                "ser NULL en la base de datos cuando un proyecto no tiene asignado un "
                "responsable. El query ORM no maneja ese caso."
            ),
        ],
    },
    {
        "user_input": "¿Qué solución se propuso para el NullPointerException en el repositorio de proyectos?",
        "reference": (
            "Se propuso añadir una condición IS NOT NULL en la cláusula WHERE de la consulta, "
            "o bien cambiar a una LEFT JOIN para manejar los proyectos sin responsable asignado. "
            "El fix fue implementado en la rama fix/project-null-assignee e incluye tests de "
            "integración para el caso edge."
        ),
        "reference_contexts": [
            (
                "Comentario en Issue #101 por Carlos López:\n"
                "La causa raíz es que 'assigned_to_id' puede ser NULL en la base de datos. "
                "Propongo añadir un IS NOT NULL en el WHERE o cambiar a una LEFT JOIN."
            ),
            (
                "Comentario en Issue #101 por Carlos López:\n"
                "Fix implementado en rama fix/project-null-assignee. "
                "Se añadió la condición IS NOT NULL y se agregaron tests de integración "
                "para el caso edge. Listo para revisión."
            ),
        ],
    },
    # ── Issue 102 ── OAuth2 con GitHub ───────────────────────────────────────
    {
        "user_input": "¿Qué tecnologías y scopes se deben usar para implementar OAuth2 con GitHub?",
        "reference": (
            "Para implementar OAuth2 con GitHub se debe usar Python 3.12, FastAPI y la "
            "librería authlib. El flujo debe solicitar los scopes 'read:user' y 'user:email'. "
            "Se recomienda el flujo Authorization Code con PKCE. Una vez obtenido el token, "
            "el perfil del usuario se crea o actualiza usando el email como identificador único."
        ),
        "reference_contexts": [
            (
                "Issue #102: Implementar autenticación OAuth2 para la integración con GitHub\n\n"
                "Se requiere implementar el flujo OAuth2 para permitir que los usuarios inicien "
                "sesión con su cuenta de GitHub. El sistema debe solicitar los scopes 'read:user' "
                "y 'user:email'. Tecnologías: Python 3.12, FastAPI, librería authlib."
            ),
            (
                "Comentario en Issue #102 por Luis Pérez:\n"
                "Revisar si authlib soporta correctamente el flujo de Authorization Code con PKCE, "
                "ya que GitHub lo recomienda para aplicaciones web."
            ),
        ],
    },
    # ── Issue 103 ── Rendimiento en Qdrant ───────────────────────────────────
    {
        "user_input": "¿Cómo se resolvió el problema de rendimiento en Qdrant con más de 50.000 documentos?",
        "reference": (
            "Se optimizaron los parámetros del índice HNSW de Qdrant: se aumentó el parámetro m "
            "de 16 (valor por defecto) a 32 y se configuró ef_construct=200. Con estos ajustes, "
            "la latencia de búsqueda con 50k documentos bajó de 8 segundos a 1.2 segundos, y la "
            "recall a top-10 mejoró de 0.87 a 0.94."
        ),
        "reference_contexts": [
            (
                "Issue #103: Degradación de rendimiento en consultas al vector store con más de 10k documentos\n\n"
                "A partir de los 10.000 documentos indexados en Qdrant, las consultas de similitud "
                "superan los 2 segundos de latencia. Se sospecha que la colección no tiene configurado "
                "un índice HNSW apropiado. El parámetro m está en el valor por defecto (16)."
            ),
            (
                "Comentario en Issue #103 por Sofía Ruiz:\n"
                "Realizamos pruebas de carga con m=32 y ef_construct=200. La latencia de búsqueda "
                "bajó de 8s a 1.2s con 50k documentos. La recall a top-10 mejoró de 0.87 a 0.94."
            ),
        ],
    },
    {
        "user_input": "¿Cuál era la latencia de consulta en Qdrant antes y después de la optimización HNSW?",
        "reference": (
            "Antes de la optimización, la latencia con 50.000 documentos era de 8 a 10 segundos. "
            "Después de ajustar m=32 y ef_construct=200 en el índice HNSW, la latencia bajó a "
            "1.2 segundos."
        ),
        "reference_contexts": [
            (
                "Issue #103: Degradación de rendimiento en consultas al vector store con más de 10k documentos\n\n"
                "Con 50k+ documentos el tiempo sube a 8-10 segundos."
            ),
            (
                "Comentario en Issue #103 por Sofía Ruiz:\n"
                "Realizamos pruebas de carga con m=32 y ef_construct=200. "
                "La latencia de búsqueda bajó de 8s a 1.2s con 50k documentos."
            ),
        ],
    },
    # ── Issue 104 ── Exportación CSV / Excel ─────────────────────────────────
    {
        "user_input": "¿Qué columnas debe tener el reporte exportable y en qué formatos está disponible?",
        "reference": (
            "El reporte exportable está disponible en formato CSV y XLSX. "
            "Las columnas requeridas son: ID, Título, Estado, Autor, Proyecto, "
            "Fecha de creación y Fecha de actualización. El CSV usa UTF-8 con BOM "
            "para compatibilidad con Excel en Windows."
        ),
        "reference_contexts": [
            (
                "Issue #104: Agregar soporte para exportar reportes en formato CSV y Excel\n\n"
                "Los usuarios solicitan poder exportar el listado de issues en formato CSV y XLSX. "
                "El CSV debe usar UTF-8 con BOM. Columnas requeridas: ID, Título, Estado, Autor, "
                "Proyecto, Fecha de creación, Fecha de actualización."
            ),
        ],
    },
    # ── Issue 105 ── CI/CD y PostgreSQL ──────────────────────────────────────
    {
        "user_input": "¿Por qué fallaban los tests de integración en GitHub Actions y cómo se solucionó?",
        "reference": (
            "Los tests fallaban porque el servicio PostgreSQL del workflow de GitHub Actions no "
            "estaba listo para aceptar conexiones cuando los tests comenzaban, produciendo el error "
            "'Connection refused: localhost:5432'. La solución fue añadir una etapa 'wait-for-postgres' "
            "que ejecuta pg_isready en loop con reintentos, y configurar el health check con "
            "'--health-cmd pg_isready --health-interval 10s --health-retries 5'."
        ),
        "reference_contexts": [
            (
                "Issue #105: Pipeline CI/CD: fallos intermitentes en la etapa de tests de integración\n\n"
                "En GitHub Actions, la etapa 'integration-tests' falla aproximadamente 1 de cada 5 "
                "ejecuciones con el error: 'Connection refused: localhost:5432'. El servicio PostgreSQL "
                "del workflow parece no estar listo cuando los tests comienzan."
            ),
            (
                "Comentario en Issue #105 por Roberto Silva:\n"
                "Solución: añadir una etapa 'wait-for-postgres' que ejecute pg_isready en un loop "
                "con reintentos antes de correr los tests. También se cambió la condición health check "
                "a 'options: --health-cmd pg_isready --health-interval 10s --health-retries 5'."
            ),
        ],
    },
]
