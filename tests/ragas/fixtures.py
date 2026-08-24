"""
tests/ragas/fixtures.py
========================
Dataset de evaluación para el RAGEngine con Ragas.

Contiene:
  - REDMINE_ISSUES_FIXTURE       : issues de Redmine sintéticos que se indexan antes de evaluar.
  - RAGAS_TESTSET                : testset base (retrocompatibilidad).
  - RAGAS_TESTSET_AUTOMERGING    : casos que requieren merge padre↔hijo (AutoMergingRetriever).
  - RAGAS_TESTSET_HYBRID         : casos con terminología léxica exacta (AutoMerging + BM25).
  - RAGAS_TESTSET_METADATA_FILTER: casos con project/status/author explícitos en la pregunta;
                                   el VectorIndexAutoRetriever infiere filtros de metadatos.

Estructura de cada entrada:
      · user_input          : pregunta del usuario
      · reference           : respuesta esperada (ground truth)
      · reference_contexts  : fragmentos del corpus que justifican la respuesta
      · bm25_hint (opcional): términos léxicos clave — solo documentación, no lo consume Ragas.
      · metadata_filter (opcional): filtros esperados — solo documentación, no lo consume Ragas.

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
    # ── Issue 106 ── Notification Service / APNs push iOS ───────────────────
    {
        "id": 106,
        "subject": "Notificaciones push no se entregan en dispositivos iOS con Background App Refresh desactivado",
        "description": (
            "Se reporta que las notificaciones push enviadas mediante APNs (Apple Push Notification service) "
            "no llegan a dispositivos iOS cuando el usuario tiene desactivado el Background App Refresh. "
            "El payload de la notificación incluye el campo 'content-available: 1' para silent notifications, "
            "que requiere Background App Refresh activo. En producción, aproximadamente el 23% de los "
            "dispositivos iOS tienen esta opción desactivada, lo que genera una pérdida significativa de alcance."
        ),
        "status": {"name": "Cerrado"},
        "author": {"name": "Paula Vega"},
        "project": {"name": "Notification Service"},
        "journals": [
            {
                "id": 207,
                "notes": (
                    "Solución implementada: enviar notificaciones de tipo 'alert' (visibles) en lugar de "
                    "silent notifications cuando el destinatario no tenga Background App Refresh activo. "
                    "El servidor ahora consulta el flag 'background_refresh_enabled' del perfil del usuario "
                    "antes de armar el payload APNs. Si está desactivado, se usa 'alert' con título y cuerpo "
                    "visibles. Los tests de regresión se agregaron en tests/integration/test_apns_delivery.py."
                ),
                "user": {"name": "Paula Vega"},
                "created_on": "2024-02-01T15:30:00Z",
            }
        ],
    },
    # ── Issue 107 ── Auth Service / JWT expirado ─────────────────────────────
    {
        "id": 107,
        "subject": "Token JWT expirado no retorna HTTP 401 sino HTTP 500 en el middleware de autenticación",
        "description": (
            "Cuando un cliente envía un token JWT expirado en el header Authorization, el middleware "
            "de autenticación lanza una excepción no controlada (JWTExpiredSignatureError) en lugar de "
            "devolver un HTTP 401 Unauthorized. El traceback muestra que la excepción no está siendo "
            "capturada en el bloque try/except del validador de tokens. Esto afecta a todas las rutas "
            "protegidas del Auth Service. Estado: En revisión por el equipo de seguridad."
        ),
        "status": {"name": "En revisión"},
        "author": {"name": "Juan Mora"},
        "project": {"name": "Auth Service"},
        "journals": [
            {
                "id": 208,
                "notes": (
                    "El bloque try/except en auth_middleware.py solo captura JWTDecodeError pero no "
                    "JWTExpiredSignatureError (subclase distinta en la librería jose). "
                    "Fix propuesto: capturar la clase base ExpiredSignatureError o ampliar el except "
                    "con verificación de tipo antes de retornar HTTP 401. "
                    "PR en revisión: fix/jwt-expired-handling."
                ),
                "user": {"name": "Juan Mora"},
                "created_on": "2024-02-05T11:00:00Z",
            }
        ],
    },
    # ── Issue 108 ── RAG Infrastructure / HNSW alta dimensionalidad ──────────
    {
        "id": 108,
        "subject": "Optimización de parámetros HNSW para colecciones con embeddings de alta dimensionalidad",
        "description": (
            "Las colecciones Qdrant que usan embeddings de 1024 dimensiones (BAAI/bge-m3) presentan "
            "mayor latencia que las de 768 dimensiones con la misma configuración HNSW. "
            "La hipótesis es que el parámetro ef (tamaño del beam search durante la consulta) necesita "
            "ajustarse proporcionalmente a la dimensionalidad. El valor actual ef=128 podría no ser "
            "suficiente para mantener una recall aceptable con vectores de 1024 dims. "
            "Responsable del análisis: Diego Fernández."
        ),
        "status": {"name": "En progreso"},
        "author": {"name": "Diego Fernández"},
        "project": {"name": "RAG Infrastructure"},
        "journals": [
            {
                "id": 209,
                "notes": (
                    "Pruebas preliminares con ef=256 muestran una mejora del 8% en recall para "
                    "embeddings de 1024 dims con colecciones de 20k vectores. "
                    "La latencia aumenta de 1.2s a 1.8s por consulta, un trade-off aceptable. "
                    "Se recomienda documentar la regla: ef >= 2 x m para mantener recall > 0.90 "
                    "independientemente de la dimensionalidad."
                ),
                "user": {"name": "Diego Fernández"},
                "created_on": "2024-02-10T09:20:00Z",
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


# ---------------------------------------------------------------------------
# Sección A — AutoMergingRetrieval
# Preguntas que requieren consolidar información del nodo hijo (comentario)
# con el nodo padre (descripción del ticket). El AutoMergingRetriever debe
# detectar que varios nodos hoja del mismo padre superan el ratio y mergear.
# ---------------------------------------------------------------------------

RAGAS_TESTSET_AUTOMERGING: list[dict] = [
    # ── Issue 101 — cobertura de descripción + 2 comentarios ─────────────────
    {
        "user_input": (
            "¿Cuáles fueron los pasos desde el reporte del error 500 en /api/projects "
            "hasta la implementación del fix en la rama fix/project-null-assignee?"
        ),
        "reference": (
            "El error 500 ocurría porque 'assigned_to_id' puede ser NULL en la BD cuando "
            "un proyecto no tiene responsable asignado, causando un NullPointerException en "
            "ProjectRepository.findByStatusAndAssignee. Carlos López reprodujo el error, "
            "identificó la causa raíz y propuso añadir IS NOT NULL o cambiar a LEFT JOIN. "
            "Finalmente el fix fue implementado en la rama fix/project-null-assignee con "
            "tests de integración para el caso edge."
        ),
        "reference_contexts": [
            (
                "Issue #101: Error 500 en el endpoint /api/projects al filtrar por estado\n\n"
                "El error ocurre cuando 'status' se combina con 'assigned_to_id'. "
                "El stack trace indica un NullPointerException en "
                "ProjectRepository.findByStatusAndAssignee."
            ),
            (
                "Comentario en Issue #101 por Carlos López:\n"
                "Reproduje el error localmente. La causa raíz es que 'assigned_to_id' puede "
                "ser NULL en la base de datos cuando un proyecto no tiene asignado un "
                "responsable. El query ORM no maneja ese caso. "
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
    # ── Issue 103 — descripción (problema) + comentario (solución) ────────────
    {
        "user_input": (
            "¿Cuál era el problema de rendimiento en Qdrant y qué parámetros HNSW "
            "resolvieron la degradación de latencia con 50.000 documentos?"
        ),
        "reference": (
            "Con más de 50.000 documentos indexados, la latencia de consulta llegaba a "
            "8-10 segundos porque el índice HNSW tenía el parámetro m en el valor por "
            "defecto (16). Al ajustar m=32 y ef_construct=200, la latencia bajó de 8s a "
            "1.2s y la recall a top-10 mejoró de 0.87 a 0.94."
        ),
        "reference_contexts": [
            (
                "Issue #103: Degradación de rendimiento en consultas al vector store con "
                "más de 10k documentos\n\n"
                "Con 50k+ documentos el tiempo sube a 8-10 segundos. "
                "El parámetro m está en el valor por defecto (16)."
            ),
            (
                "Comentario en Issue #103 por Sofía Ruiz:\n"
                "Realizamos pruebas de carga con m=32 y ef_construct=200. "
                "La latencia bajó de 8s a 1.2s con 50k documentos. "
                "La recall a top-10 mejoró de 0.87 a 0.94."
            ),
        ],
    },
    # ── Issue 104 — descripción (especificaciones) + comentario (librería) ────
    {
        "user_input": (
            "¿Qué especificaciones técnicas tiene el módulo de exportación de reportes "
            "y qué librería Python se usa para generar los archivos XLSX?"
        ),
        "reference": (
            "El módulo exporta reportes en formato CSV y XLSX. El CSV usa UTF-8 con BOM "
            "para compatibilidad con Excel en Windows. Para XLSX se usa la librería "
            "openpyxl, con encabezados en negrita y autofit de columnas. "
            "Las columnas son: ID, Título, Estado, Autor, Proyecto, Fecha de creación "
            "y Fecha de actualización. Los tests unitarios están en tests/unit/test_export.py."
        ),
        "reference_contexts": [
            (
                "Issue #104: Agregar soporte para exportar reportes en formato CSV y Excel\n\n"
                "El CSV debe usar UTF-8 con BOM para compatibilidad con Excel en Windows. "
                "El XLSX debe incluir encabezados en negrita y autofit. "
                "Columnas requeridas: ID, Título, Estado, Autor, Proyecto, "
                "Fecha de creación, Fecha de actualización."
            ),
            (
                "Comentario en Issue #104 por Elena Martínez:\n"
                "Usando la librería openpyxl para el XLSX y el módulo csv estándar de Python "
                "para el CSV. El BOM se añade escribiendo b'\\xef\\xbb\\xbf' al inicio del archivo. "
                "Los tests unitarios están en tests/unit/test_export.py."
            ),
        ],
    },
    # ── Issue 105 — descripción (problema) + comentario (health check final) ──
    {
        "user_input": (
            "¿Por qué fallaban los tests de integración en GitHub Actions y cuál fue "
            "la configuración definitiva del health check para PostgreSQL?"
        ),
        "reference": (
            "Los tests fallaban porque PostgreSQL no estaba listo cuando comenzaban, "
            "generando 'Connection refused: localhost:5432'. La solución fue agregar "
            "la etapa 'wait-for-postgres' que ejecuta pg_isready en loop con reintentos, "
            "y configurar el health check con "
            "'--health-cmd pg_isready --health-interval 10s --health-retries 5'."
        ),
        "reference_contexts": [
            (
                "Issue #105: Pipeline CI/CD: fallos intermitentes en la etapa de tests de integración\n\n"
                "La etapa 'integration-tests' falla con: 'Connection refused: localhost:5432'. "
                "El servicio PostgreSQL no está listo cuando los tests comienzan."
            ),
            (
                "Comentario en Issue #105 por Roberto Silva:\n"
                "Solución: añadir una etapa 'wait-for-postgres' que ejecute pg_isready en loop "
                "con reintentos. Se cambió la condición health check a "
                "'options: --health-cmd pg_isready --health-interval 10s --health-retries 5'."
            ),
        ],
    },
]


# ---------------------------------------------------------------------------
# Sección B — Hybrid (AutoMerging + BM25)
# Preguntas con terminología técnica léxica muy específica: nombres de método,
# comandos exactos, siglas y flags. BM25 aporta signal en la rama sparse.
# La clave 'bm25_hint' documenta los términos clave (no la consume Ragas).
# ---------------------------------------------------------------------------

RAGAS_TESTSET_HYBRID: list[dict] = [
    # ── Issue 101 — nombre de método + cláusula SQL exacta ───────────────────
    {
        "user_input": (
            "¿En qué método exacto de ProjectRepository ocurre el NullPointerException "
            "y qué cláusula SQL IS NOT NULL se añadió para resolverlo?"
        ),
        "reference": (
            "El NullPointerException ocurre en el método "
            "ProjectRepository.findByStatusAndAssignee. "
            "La solución fue añadir una condición IS NOT NULL en la cláusula WHERE de la "
            "consulta ORM, o bien cambiar a una LEFT JOIN para manejar los registros con "
            "assigned_to_id NULL."
        ),
        "reference_contexts": [
            (
                "Issue #101: Error 500 en el endpoint /api/projects al filtrar por estado\n\n"
                "El stack trace indica un NullPointerException en la capa de repositorio "
                "(ProjectRepository.findByStatusAndAssignee)."
            ),
            (
                "Comentario en Issue #101 por Carlos López:\n"
                "Propongo añadir un IS NOT NULL en el WHERE o cambiar a una LEFT JOIN."
            ),
        ],
        "bm25_hint": "NullPointerException, ProjectRepository.findByStatusAndAssignee, IS NOT NULL",
    },
    # ── Issue 105 — comando pg_isready + flags exactos ───────────────────────
    {
        "user_input": (
            "¿Qué comando pg_isready se usa en el workflow y cuáles son las opciones "
            "--health-interval y --health-retries configuradas en el health check?"
        ),
        "reference": (
            "Se usa el comando pg_isready en un loop con reintentos en la etapa "
            "'wait-for-postgres'. El health check del servicio PostgreSQL se configura con: "
            "--health-cmd pg_isready, --health-interval 10s y --health-retries 5."
        ),
        "reference_contexts": [
            (
                "Comentario en Issue #105 por Roberto Silva:\n"
                "Solución: añadir una etapa 'wait-for-postgres' que ejecute pg_isready en un "
                "loop con reintentos antes de correr los tests. También se cambió la condición "
                "health check a 'options: --health-cmd pg_isready --health-interval 10s "
                "--health-retries 5'."
            ),
        ],
        "bm25_hint": "pg_isready, --health-cmd, --health-interval 10s, --health-retries 5",
    },
    # ── Issue 106 — APNs + content-available + background_refresh_enabled ─────
    {
        "user_input": (
            "¿Qué tipo de payload APNs usa el Notification Service para silent notifications "
            "y cuál es el flag background_refresh_enabled del perfil del usuario?"
        ),
        "reference": (
            "Las silent notifications usan el campo 'content-available: 1' en el payload APNs, "
            "que requiere Background App Refresh activo en el dispositivo iOS. "
            "El servidor consulta el flag 'background_refresh_enabled' del perfil del usuario: "
            "si está desactivado, envía notificaciones de tipo 'alert' (visibles) en su lugar."
        ),
        "reference_contexts": [
            (
                "Issue #106: Notificaciones push no se entregan en dispositivos iOS con "
                "Background App Refresh desactivado\n\n"
                "El payload incluye el campo 'content-available: 1' para silent notifications, "
                "que requiere Background App Refresh activo. "
                "El 23% de los dispositivos iOS tienen esta opción desactivada."
            ),
            (
                "Comentario en Issue #106 por Paula Vega:\n"
                "El servidor ahora consulta el flag 'background_refresh_enabled' del perfil del "
                "usuario antes de armar el payload APNs. "
                "Si está desactivado, se usa 'alert' con título y cuerpo visibles."
            ),
        ],
        "bm25_hint": "APNs, content-available, background_refresh_enabled, Background App Refresh",
    },
    # ── Issue 102 — scopes OAuth2 + PKCE + authlib ───────────────────────────
    {
        "user_input": (
            "¿Qué scopes OAuth2 exactos solicita el sistema a GitHub y qué flujo de "
            "Authorization Code con PKCE soporta la librería authlib?"
        ),
        "reference": (
            "El sistema solicita los scopes 'read:user' y 'user:email' de GitHub. "
            "Se recomienda usar el flujo Authorization Code con PKCE, ya que GitHub lo "
            "recomienda para aplicaciones web, y la librería authlib lo soporta."
        ),
        "reference_contexts": [
            (
                "Issue #102: Implementar autenticación OAuth2 para la integración con GitHub\n\n"
                "El sistema debe solicitar los scopes 'read:user' y 'user:email'. "
                "Tecnologías: Python 3.12, FastAPI, librería authlib."
            ),
            (
                "Comentario en Issue #102 por Luis Pérez:\n"
                "Revisar si authlib soporta correctamente el flujo de Authorization Code con PKCE, "
                "ya que GitHub lo recomienda para aplicaciones web."
            ),
        ],
        "bm25_hint": "read:user, user:email, Authorization Code, PKCE, authlib",
    },
]


# ---------------------------------------------------------------------------
# Sección C — AutoMerging + Hybrid + Metadata Filtering
# Preguntas que mencionan explícitamente project, status o author en el texto.
# El VectorIndexAutoRetriever infiere los filtros de metadatos a partir del
# lenguaje natural de la pregunta (no se usan MetadataFilters hardcodeados).
# La clave 'metadata_filter' documenta los filtros esperados (no la consume Ragas).
# ---------------------------------------------------------------------------

RAGAS_TESTSET_METADATA_FILTER: list[dict] = [
    # ── Filter: project="Auth Service" ───────────────────────────────────────
    {
        "user_input": (
            "En el proyecto Auth Service, ¿qué issue trata sobre la implementación "
            "del flujo OAuth2 con GitHub y qué tecnologías usa?"
        ),
        "reference": (
            "El Issue #102 del proyecto Auth Service cubre la implementación del flujo "
            "OAuth2 con GitHub. Usa Python 3.12, FastAPI y la librería authlib, solicita "
            "los scopes 'read:user' y 'user:email', y recomienda el flujo "
            "Authorization Code con PKCE."
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
        "metadata_filter": {"project": "Auth Service"},
    },
    # ── Filter: project="Auth Service" + status="En revisión" ─────────────────
    {
        "user_input": (
            "¿Qué issue del proyecto Auth Service se encuentra en estado 'En revisión' "
            "relacionado con tokens JWT y qué excepción no se estaba capturando?"
        ),
        "reference": (
            "El Issue #107 del proyecto Auth Service, en estado 'En revisión', describe "
            "que un token JWT expirado no retorna HTTP 401 sino HTTP 500. La causa es que "
            "el middleware solo captura JWTDecodeError pero no JWTExpiredSignatureError. "
            "El fix propuesto está en el PR fix/jwt-expired-handling."
        ),
        "reference_contexts": [
            (
                "Issue #107: Token JWT expirado no retorna HTTP 401 sino HTTP 500 "
                "en el middleware de autenticación\n\n"
                "Cuando un cliente envía un token JWT expirado, el middleware lanza "
                "JWTExpiredSignatureError en lugar de devolver HTTP 401 Unauthorized. "
                "Estado: En revisión por el equipo de seguridad."
            ),
            (
                "Comentario en Issue #107 por Juan Mora:\n"
                "El bloque try/except solo captura JWTDecodeError pero no "
                "JWTExpiredSignatureError (subclase distinta en la librería jose). "
                "Fix propuesto: capturar la clase base ExpiredSignatureError. "
                "PR en revisión: fix/jwt-expired-handling."
            ),
        ],
        "metadata_filter": {"project": "Auth Service", "status": "En revisión"},
    },
    # ── Filter: author="Diego Fernández" + project="RAG Infrastructure" ───────
    {
        "user_input": (
            "¿Qué issues del proyecto RAG Infrastructure fueron reportados por Diego Fernández "
            "y cuáles son sus conclusiones sobre la optimización de parámetros HNSW?"
        ),
        "reference": (
            "Diego Fernández reportó el Issue #103 (degradación de latencia con m=16 por defecto) "
            "y el Issue #108 (ef=128 insuficiente para embeddings de 1024 dims). "
            "En #103 concluyó que m=32 y ef_construct=200 bajan la latencia de 8s a 1.2s y "
            "mejoran la recall de 0.87 a 0.94. En #108 encontró que ef=256 mejora el recall "
            "un 8% y recomienda la regla ef >= 2 x m para recall > 0.90."
        ),
        "reference_contexts": [
            (
                "Issue #103: Degradación de rendimiento en consultas al vector store "
                "con más de 10k documentos\n\n"
                "Con 50k+ documentos el tiempo sube a 8-10 segundos. "
                "El parámetro m está en el valor por defecto (16). "
                "Autor: Diego Fernández."
            ),
            (
                "Issue #108: Optimización de parámetros HNSW para colecciones con embeddings "
                "de alta dimensionalidad\n\n"
                "Las colecciones con embeddings de 1024 dimensiones presentan mayor latencia. "
                "El valor actual ef=128 podría no ser suficiente. "
                "Responsable del análisis: Diego Fernández."
            ),
            (
                "Comentario en Issue #108 por Diego Fernández:\n"
                "Pruebas con ef=256 muestran mejora del 8% en recall para embeddings de 1024 dims. "
                "Se recomienda la regla: ef >= 2 x m para mantener recall > 0.90."
            ),
        ],
        "metadata_filter": {"author": "Diego Fernández", "project": "RAG Infrastructure"},
    },
    # ── Filter: project="Notification Service" + status="Cerrado" ─────────────
    {
        "user_input": (
            "En el proyecto Notification Service, ¿qué problema de entrega de notificaciones "
            "push en iOS fue reportado con estado Cerrado y cuál fue la solución?"
        ),
        "reference": (
            "El Issue #106 del proyecto Notification Service (estado: Cerrado) reporta que "
            "las notificaciones push con payload 'content-available: 1' no llegan a dispositivos "
            "iOS con Background App Refresh desactivado (el 23% de los dispositivos). "
            "La solución fue consultar el flag 'background_refresh_enabled' del usuario y usar "
            "notificaciones tipo 'alert' cuando está desactivado."
        ),
        "reference_contexts": [
            (
                "Issue #106: Notificaciones push no se entregan en dispositivos iOS con "
                "Background App Refresh desactivado\n\n"
                "El payload 'content-available: 1' requiere Background App Refresh activo. "
                "El 23% de los dispositivos iOS tienen esta opción desactivada."
            ),
            (
                "Comentario en Issue #106 por Paula Vega:\n"
                "Solución implementada: el servidor consulta el flag 'background_refresh_enabled' "
                "del usuario y usa notificaciones 'alert' con título y cuerpo visibles cuando "
                "está desactivado."
            ),
        ],
        "metadata_filter": {"project": "Notification Service", "status": "Cerrado"},
    },
]
