#!/usr/bin/env python3
"""
src/redmine/generate_realistic_tickets.py
=========================================
Script para poblar Redmine con casos de prueba altamente realistas, extensos y técnicos
para evaluar el agente LangGraph (clarificación, deduplicación, creación) y el RAGEngine.

Los tickets están organizados en los proyectos existentes:
  - infraestructura-de-servicios : SRE, DevOps, bases de datos, redes, K8s, observabilidad.
  - proyecto-prueba             : APIs backend, lógica de negocio, frontend web, auth, integraciones.

Uso:
    python src/redmine/generate_realistic_tickets.py [--dry-run] [--index]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()


def get_realistic_tickets_dataset() -> list[dict[str, Any]]:
    """
    Retorna el catálogo completo de tickets técnicos realistas, extensos y estructurados.
    Cada elemento contiene metadatos completos, descripción formateada en Textile/Markdown
    y notas/journals que representan el ciclo de vida del incidente.
    """
    return [
        # ==============================================================================
        # PROYECTO: infraestructura-de-servicios
        # ==============================================================================
        {
            "project_id": "infraestructura-de-servicios",
            "subject": "[PostgreSQL] Conexiones agotadas (FATAL: sorry, too many clients already) en el pool de pgBouncer",
            "tracker_id": 1,  # Bug
            "status_id": 2,  # En curso
            "priority_id": 4,  # Urgente
            "description": """h2. Descripción del Problema
La base de datos principal de producción (PostgreSQL v15.4 alojada en AWS RDS) comenzó a rechazar conexiones entrantes a las 04:12 UTC. Las aplicaciones conectadas vía pgBouncer reportan fallos masivos de timeout y errores 500 en cascada.

h2. Entorno y Componentes Afectados
* *Servicio*: PostgreSQL Cluster Primario (db-prod-primary.internal)
* *Pooler*: pgBouncer v1.21 corriendo como DaemonSet en Kubernetes
* *Parámetros actuales*: @max_connections = 300@, @default_pool_size = 20@, @pool_mode = transaction@

h2. Evidencia Técnica y Logs
Fragmento del log de PostgreSQL (/var/log/postgresql/postgresql.log):
```log
2024-03-10 04:12:01.324 UTC [28194] FATAL: sorry, too many clients already
2024-03-10 04:12:01.329 UTC [28195] LOG: could not fork new process for connection: Cannot allocate memory
2024-03-10 04:12:02.102 UTC [28201] FATAL: sorry, too many clients already
```

Métricas de pgBouncer (@SHOW POOLS@):
```text
database | user | cl_active | cl_waiting | sv_active | sv_idle | sv_used | maxwait
db_prod  | app  |       298 |        142 |       295 |       0 |       5 |      48
```

h2. Pasos para Reproducir
1. Ejecutar el cronjob de consolidación nocturna de balances (@job-nightly-reconcile@).
2. Monitorear el número de clientes activos con @SELECT count(*) FROM pg_stat_activity;@.
3. Observar cómo el número de conexiones idle en transacción trepa hasta saturar el límite fijado de 300 conexiones.

h2. Impacto en el Negocio
Degradación crítica: Todos los microservicios dependientes de la base de datos están arrojando errores de persistencia. El checkout de la plataforma se encuentra inoperativo.""",
            "notes": [
                "Investigación inicial: El microservicio de conciliación bancaria disparó 50 hilos concurrentes que abren transacciones JDBC largas sin cerrarlas adecuadamente en el bloque finally.",
                "Mitigación temporal: Se elevó temporalmente max_connections a 450 en el parámetro group de RDS y se reinició pgBouncer para purgar clientes colgados.",
                "Se acordó con el equipo de backend implementar timeouts de transacción estrictos: idle_in_transaction_session_timeout = '60s'.",
            ],
        },
        {
            "project_id": "infraestructura-de-servicios",
            "subject": "[Kubernetes] OOMKilled recurrente en pods de Celery Worker durante exportación masiva de reportes",
            "tracker_id": 1,
            "status_id": 2,
            "priority_id": 3,  # Alta
            "description": """h2. Descripción del Problema
Los pods que ejecutan los workers asíncronos de Celery (@celery-worker-reports@) están siendo terminados sistemáticamente por el kernel con señal OOMKilled (Exit Code 137). El problema ocurre cuando múltiples usuarios solicitan la exportación de métricas anuales en formato CSV o PDF.

h2. Entorno y Componentes Afectados
* *Clúster*: EKS v1.28
* *Namespace*: production-workers
* *Imagen Docker*: techvanguard/worker-reports:v2.4.1
* *Límites de recursos*: requests: 512Mi / limits: 1536Mi

h2. Logs del Pod y Eventos de Kubernetes
Evento registrado en @kubectl describe pod celery-worker-reports-7b8f9c-4x2lz@:
```yaml
Last State:     Terminated
  Reason:       OOMKilled
  Exit Code:    137
  Started:      Sun, 10 Mar 2024 10:15:20 -0300
  Finished:     Sun, 10 Mar 2024 10:32:45 -0300
```

Log capturado justo antes del reinicio:
```log
[2024-03-10 10:32:40,112: INFO/ForkPoolWorker-2] Task tasks.generate_annual_audit_report[9a12e3] received
[2024-03-10 10:32:44,550: DEBUG/ForkPoolWorker-2] Loading 450,000 transaction records into memory Pandas DataFrame
Killed
```

h2. Pasos para Reproducir
1. Encolar una tarea de generación de reporte que consulte más de 100.000 filas.
2. Monitorear el consumo de memoria del pod con @kubectl top pod -l app=celery-worker-reports@.
3. Observar la curva lineal de crecimiento de memoria hasta alcanzar los 1.5GB y la posterior muerte del contenedor.

h2. Impacto en el Negocio
Los reportes financieros quedan en estado pendiente indefinido y los clientes corporativos no pueden descargar sus auditorías fiscales.""",
            "notes": [
                "Se confirmó que la tarea lee todo el conjunto de resultados con pandas.read_sql_query sin utilizar el parámetro chunksize.",
                "Se abrió una rama en desarrollo para refactorizar la exportación en streaming hacia un bucket S3 con multipart upload.",
            ],
        },
        {
            "project_id": "infraestructura-de-servicios",
            "subject": "[Nginx] Error 502 Bad Gateway intermitente por upstream timeout en endpoints de streaming gRPC",
            "tracker_id": 1,
            "status_id": 1,  # Nueva
            "priority_id": 3,
            "description": """h2. Descripción del Problema
Los clientes que consumen la API de telemetría en tiempo real a través del Ingress de Nginx experimentan caídas de conexión HTTP 502 Bad Gateway exactamente a los 60 segundos de haber iniciado el stream bidireccional.

h2. Entorno y Componentes Afectados
* *Ingress Controller*: ingress-nginx v1.9.4 en Kubernetes
* *Protocolo*: HTTP/2 gRPC (@grpc-web@)
* *Servicio de destino*: telemetry-collector.production.svc.cluster.local:50051

h2. Logs y Métricas
Logs en Nginx Ingress:
```log
2024-03-11T14:22:15+00:00 [error] 1422#1422: *89124 upstream prematurely closed connection while reading response header from upstream, client: 190.210.45.12, server: telemetry.techvanguard.com, request: "POST /telemetry.StreamService/Subscribe HTTP/2.0", upstream: "grpc://10.244.3.44:50051"
```

h2. Diagnóstico Preliminar
El ingress controller no tiene configuradas las anotaciones de timeout de lectura gRPC. El valor por defecto de @proxy_read_timeout@ es de 60s, lo que corta cualquier canal gRPC que no envíe un ping periódico.

h2. Pasos para Reproducir
1. Conectarse al endpoint con @grpcurl -vv -d '{"client_id": "test"}' telemetry.techvanguard.com:443 telemetry.StreamService/Subscribe@.
2. Mantener la conexión abierta sin enviar tráfico por 60 segundos.
3. Recibir el error @transport: Error while dialing: 502 Bad Gateway@.""",
            "notes": [
                "Se requiere añadir las anotaciones: nginx.ingress.kubernetes.io/proxy-read-timeout: '3600' y habilitar keepalive en el backend.",
            ],
        },
        {
            "project_id": "infraestructura-de-servicios",
            "subject": "[OpenVPN] Fallo de handshake TLS (TLS error: TLS object -> incoming plaintext read error) por CA vencida",
            "tracker_id": 3,  # Soporte
            "status_id": 3,  # Resuelta
            "priority_id": 4,  # Urgente
            "description": """h2. Descripción del Problema
Varios ingenieros reportan imposibilidad total para conectarse a la red interna mediante el cliente OpenVPN en distribuciones Linux y macOS. El cliente se queda en bucle de reconexión.

h2. Evidencia y Logs del Cliente
```log
Mon Mar  4 09:12:33 2024 TLS Error: TLS key negotiation failed to occur within 60 seconds (check your network connectivity)
Mon Mar  4 09:12:33 2024 TLS Error: TLS handshake failed
Mon Mar  4 09:12:33 2024 VERIFY ERROR: depth=1, error=certificate has expired: C=US, O=TechVanguard, CN=TechVanguard Intermediate CA
```

h2. Causa Raíz Identificada
El certificado de la CA Intermedia venció el 3 de marzo a las 23:59 UTC.

h2. Solución Aplicada (Documentada)
1. Se generó un nuevo par de certificados CA Intermedia con vigencia de 3 años en el servidor EasyRSA.
2. Se publicó el nuevo archivo de configuración @techvanguard-corp-v2.ovpn@ en el portal interno de IT (https://portal.techvanguard.internal/vpn).
3. Los usuarios deben descargar el nuevo bundle y actualizar su certificado de cliente local, verificando que su reloj NTP esté sincronizado con @chronyd@.""",
            "notes": [
                "Certificado renovado y desplegado en los servidores VPN gateway de US-East y EU-West.",
                "Se verificó la conexión exitosa de 45 usuarios remotos. Incidencia resuelta.",
            ],
        },
        {
            "project_id": "infraestructura-de-servicios",
            "subject": "[Redis] Fuga de descriptores de sockets en estado CLOSE_WAIT provocando bloqueo de Sentinel",
            "tracker_id": 1,
            "status_id": 2,
            "priority_id": 3,
            "description": """h2. Descripción del Problema
El nodo primario de Redis Cluster (redis-cache-01) deja de atender peticiones de lectura y escritura cada 72 horas. El proceso @redis-server@ sigue activo pero no responde a pings de Sentinel ni a comandos @INFO@.

h2. Diagnóstico Técnico
Al inspeccionar los sockets con @lsof -p $(pgrep redis-server)@, se observan más de 65.000 descriptores en estado @CLOSE_WAIT@ originados por conexiones cortadas abruptamente desde los pods de nodejs.
```bash
$ netstat -tonp | grep redis-server | grep CLOSE_WAIT | wc -l
64892
```

h2. Pasos a Seguir
1. Habilitar tcp-keepalive en redis.conf (@tcp-keepalive 60@).
2. Revisar la configuración del cliente ioredis en el backend para asegurar que maneje correctamente el evento error y cierre los sockets en desconexión.""",
            "notes": [
                "Se aplicó 'CONFIG SET tcp-keepalive 60' en caliente. La cantidad de sockets en CLOSE_WAIT se redujo a menos de 50.",
            ],
        },
        {
            "project_id": "infraestructura-de-servicios",
            "subject": "[Docker] Saturación de disco al 100% en /var/lib/docker/overlay2 por falta de rotación de logs json-file",
            "tracker_id": 1,
            "status_id": 3,  # Resuelta
            "priority_id": 4,
            "description": """h2. Descripción del Problema
El nodo worker k8s-node-compute-04 se marcó en estado @NotReady@ por @DiskPressure@. El filesystem raíz montado en @/dev/nvme0n1p1@ alcanzó el 100% de uso (500GB ocupados).

h2. Análisis de Directorios
```bash
# du -sh /var/lib/docker/containers/* | sort -h | tail -n 5
48G     /var/lib/docker/containers/a12f8...
72G     /var/lib/docker/containers/b839c...
180G    /var/lib/docker/containers/c949a... (app-logging-daemon-json.log)
```

h2. Solución Aplicada
Se configuró el daemon de Docker (/etc/docker/daemon.json) con directivas de rotación de logs por defecto:
```json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "50m",
    "max-file": "3"
  }
}
```
Se ejecutó @systemctl reload docker@ y se liberaron 320GB de espacio mediante script de truncado seguro.""",
            "notes": [
                "Solución aplicada con Ansible en todos los 18 nodos del clúster. Alerta de Datadog normalizada.",
            ],
        },
        {
            "project_id": "infraestructura-de-servicios",
            "subject": "[Traefik] Certificado TLS wildcard (*.techvanguard.internal) próximo a expirar en clúster productivo",
            "tracker_id": 2,  # Tarea
            "status_id": 1,
            "priority_id": 4,
            "description": """h2. Descripción
El certificado emitido por HashiCorp Vault para la zona interna @*.techvanguard.internal@ tiene fecha de caducidad en 7 días calendario (17 de marzo de 2024).

h2. Tareas a Ejecutar
1. Solicitar la renovación automática mediante el cert-manager de Kubernetes apuntando al ClusterIssuer de Vault.
2. Verificar que el Secret @wildcard-internal-tls@ se actualice en todos los namespaces compartidos (@core@, @staging@, @production@).
3. Validar handshake en los endpoints internos con @curl -vI https://api.techvanguard.internal/health@.""",
            "notes": [
                "Se verificó que cert-manager falló en el último intento por un cambio en la política AppRole de Vault.",
            ],
        },
        {
            "project_id": "infraestructura-de-servicios",
            "subject": "[Calico] NetworkPolicy bloquea tráfico saliente en puerto 5432 desde el namespace de staging",
            "tracker_id": 1,
            "status_id": 3,
            "priority_id": 3,
            "description": """h2. Descripción del Problema
Tras la actualización del CNI Calico a v3.27, todos los pods en el namespace @staging@ perdieron visibilidad hacia la base de datos PostgreSQL alojada en el segmento de red de infraestructura (10.0.12.50:5432).

h2. Causa
La política por defecto @default-deny-egress@ no incluía la excepción para el bloque CIDR de las bases de datos internas.

h2. Solución
Se aplicó el manifiesto:
```yaml
apiVersion: projectcalico.org/v3
kind: NetworkPolicy
metadata:
  name: allow-db-egress
  namespace: staging
spec:
  egress:
    - action: Allow
      protocol: TCP
      destination:
        nets:
          - 10.0.12.0/24
        ports:
          - 5432
```""",
            "notes": [
                "Verificada la conectividad mediante 'nc -zv 10.0.12.50 5432' desde dentro del pod de staging.",
            ],
        },
        {
            "project_id": "infraestructura-de-servicios",
            "subject": "[Kafka] Desbalanceo de particiones y lag crítico (>50k mensajes) en tópico events.transactions",
            "tracker_id": 1,
            "status_id": 2,
            "priority_id": 4,
            "description": """h2. Descripción del Problema
El consumidor del microservicio de contabilidad reporta un lag de más de 50.000 mensajes en la partición 2 del tópico @events.transactions@. Las particiones 0 y 1 tienen lag cero.

h2. Métricas de Kafka
```text
GROUP                  TOPIC               PARTITION  CURRENT-OFFSET  LOG-END-OFFSET  LAG
accounting-consumer    events.transactions 0          482910          482910          0
accounting-consumer    events.transactions 1          491023          491023          0
accounting-consumer    events.transactions 2          312001          365412          53411
```

h2. Causa
El productor está particionando usando la clave @user_id@, y un único cliente institucional de gran volumen está concentrando el 80% del tráfico transaccional en la misma partición.""",
            "notes": [
                "Se aumentará el número de particiones de 3 a 12 y se modificará la clave de particionado a 'user_id + transaction_uuid'.",
            ],
        },
        {
            "project_id": "infraestructura-de-servicios",
            "subject": "[Qdrant] Búsqueda vectorial lenta (>3.5s) en colección de issues por falta de optimización HNSW",
            "tracker_id": 1,
            "status_id": 2,
            "priority_id": 3,
            "description": """h2. Descripción del Problema
Las consultas semánticas al motor de búsqueda vectorial Qdrant superan los 3.5 segundos de latencia P99 cuando la colección @redmine_issues@ sobrepasó los 25.000 vectores (dimensión 1024 bge-m3).

h2. Configuración Actual
* @hnsw_config.m = 16@
* @hnsw_config.ef_construct = 100@
* Los vectores están almacenados en disco sin memmap habilitado.

h2. Propuesta de Optimización
1. Incrementar @m = 32@ y @ef_construct = 256@.
2. Habilitar @on_disk_payload = true@ y mantener los vectores indexados en RAM mediante memoria compartida.""",
            "notes": [
                "Se recreó la colección de pruebas con los parámetros optimizados: la latencia de recuperación bajó de 3.5s a 140ms.",
            ],
        },
        {
            "project_id": "infraestructura-de-servicios",
            "subject": "[CI/CD] Fallos por falta de memoria compartida /dev/shm en runners de GitLab al ejecutar Cypress",
            "tracker_id": 1,
            "status_id": 3,
            "priority_id": 2,  # Normal
            "description": """h2. Descripción del Problema
Los tests end-to-end de frontend fallaban aleatoriamente con @Chrome crashed: Out of memory@ en los runners de GitLab basados en Docker-in-Docker.

h2. Solución
Se montó un volumen @shm_size: '2g'@ en la configuración del runner (/etc/gitlab-runner/config.toml) y se añadió el flag @--disable-dev-shm-usage@ en el script de lanzamiento de Chrome.""",
            "notes": [
                "Pipelines de CI estabilizados con 100% de pasaje en las últimas 40 ejecuciones.",
            ],
        },
        {
            "project_id": "infraestructura-de-servicios",
            "subject": "[PostgreSQL] Streaming replication lag supera los 45 minutos hacia la réplica de solo lectura",
            "tracker_id": 1,
            "status_id": 1,
            "priority_id": 3,
            "description": """h2. Síntomas
La réplica de lectura utilizada por el equipo de Data Analytics tiene un retraso de 45 minutos respecto al primario. La métrica @pg_wal_lsn_diff@ muestra 8.5 GB de diferencia.

h2. Diagnóstico
Queries analíticas de lectura de larga duración están bloqueando la aplicación de registros WAL con el mensaje @canceling statement due to conflict with recovery@. Se debe evaluar activar @hot_standby_feedback = on@.""",
            "notes": [],
        },
        {
            "project_id": "infraestructura-de-servicios",
            "subject": "[Seguridad] Vulnerabilidad crítica CVE-2024-21626 (runc container escape) en nodos de Kubernetes",
            "tracker_id": 1,
            "status_id": 3,
            "priority_id": 5,  # Inmediata
            "description": """h2. Descripción
Se publicó la vulnerabilidad crítica CVE-2024-21626 (Leaky Vessels) que permite el escape de contenedores a nivel de host mediante descriptores de archivos filtrados en @runc@ (< v1.1.12).

h2. Acciones Ejecutadas
1. Drenado secuencial de nodos k8s (@kubectl drain --ignore-daemonsets@).
2. Actualización de paquetes @containerd.io@ y @runc@ a v1.1.12 en todos los nodos Ubuntu 22.04 LTS.
3. Desmarcado de nodos (@kubectl uncordon@) y re-escaneo con Trivy.""",
            "notes": [
                "Parche desplegado en producción sin interrupción de servicios gracias al pod disruption budget.",
            ],
        },
        {
            "project_id": "infraestructura-de-servicios",
            "subject": "[AWS VPN] Túnel IPSec Site-to-Site presenta caídas por fragmentación de paquetes y desfase de MTU",
            "tracker_id": 1,
            "status_id": 2,
            "priority_id": 3,
            "description": """h2. Síntomas
Transferencias de archivos SCP superiores a 50MB entre la oficina on-premise y AWS VPC se congelan en el 99%. Las peticiones HTTP simples funcionan con normalidad.

h2. Causa
El túnel IPSec encapsula paquetes agregando cabeceras ESP (56 bytes extra), excediendo la MTU estándar de 1500 bytes del enlace físico y provocando descarte por DF (Don't Fragment) bit activado.

h2. Fix Propuesto
Configurar MSS Clamping en el router de borde: @iptables -t mangle -A FORWARD -p tcp --tcp-flags SYN,RST SYN -j TCPMSS --clamp-mss-to-pmtu@ o fijar MTU a 1420 en la interfaz vti0.""",
            "notes": [
                "Probado con ping con tamaño de payload: 'ping -M do -s 1472 10.0.1.1' confirmó la pérdida de paquetes.",
            ],
        },
        {
            "project_id": "infraestructura-de-servicios",
            "subject": "[PostgreSQL] Deadlock recurrente entre transacciones concurrentes de actualización de stock y facturación",
            "tracker_id": 1,
            "status_id": 1,
            "priority_id": 3,
            "description": """h2. Evidencia de Logs
```log
ERROR: deadlock detected
DETAIL: Process 18402 waits for ShareLock on transaction 8912401; blocked by process 18410.
Process 18410 waits for ExclusiveLock on tuple (412, 18) of relation "inventory_items"; blocked by process 18402.
HINT: See server log for query details.
```

h2. Causa
El servicio de órdenes adquiere bloqueos en orden @orders -> inventory_items@, mientras que el servicio de devoluciones adquiere bloqueos en orden inverso @inventory_items -> orders@. Se requiere unificar el orden de adquisición de locks.""",
            "notes": [],
        },
        {
            "project_id": "infraestructura-de-servicios",
            "subject": "[Elasticsearch] Clúster pasa a estado RED por fragmentación de shards no asignados tras reinicio de nodo",
            "tracker_id": 1,
            "status_id": 3,
            "priority_id": 4,
            "description": """h2. Incidente
El clúster de logs de Elasticsearch reportó estado RED. 4 shards primarios del índice @fluentbit-2024.03.01@ quedaron en estado @UNASSIGNED@ debido a corrupción de metadata tras un reinicio forzado del nodo es-data-02.

h2. Recuperación
Se ejecutó comando de asignación forzada de réplicas y re-enrutado con @POST /_cluster/reroute@, recuperando el 99.8% de los logs sin pérdida de datos críticos.""",
            "notes": [
                "Clúster recuperó estado GREEN tras 40 minutos de replicación interna.",
            ],
        },
        # ==============================================================================
        # PROYECTO: proyecto-prueba
        # ==============================================================================
        {
            "project_id": "proyecto-prueba",
            "subject": "[Checkout API] Error 500 Internal Server Error por KeyError: 'billing_address' en POST /api/v1/checkout",
            "tracker_id": 1,
            "status_id": 2,
            "priority_id": 4,  # Urgente
            "description": """h2. Descripción del Problema
Cuando un cliente registrado intenta completar una compra seleccionando la opción 'Misma dirección de envío para facturación', la API responde con HTTP 500 Internal Server Error.

h2. Endpoint Afectado
@POST https://api.techvanguard.com/api/v1/checkout@

h2. Payload de Reproducción
```json
{
  "cart_id": "cart_991823a",
  "shipping_address": {
    "street": "Av. Corrientes 1234",
    "city": "CABA",
    "country": "AR"
  },
  "use_shipping_for_billing": true
}
```

h2. Stacktrace Capturado
```python
Traceback (most recent call last):
  File "/app/services/checkout_service.py", line 142, in process_order
    billing_data = payload["billing_address"]
KeyError: 'billing_address'
```

h2. Causa Raíz
El validador de Pydantic asume que @billing_address@ siempre viene presente si no se activa el flag de resolución condicional en el esquema @CheckoutRequestSchema@.

h2. Impacto en el Negocio
Caída del 28% en la tasa de conversión durante el fin de semana. Afecta a usuarios móviles.""",
            "notes": [
                "Reproducido en entorno de staging. Se necesita un root_validator en el modelo de Pydantic.",
                "PR #512 abierto con el test de regresión correspondiente.",
            ],
        },
        {
            "project_id": "proyecto-prueba",
            "subject": "[Keycloak] Error 'invalid_grant' en flujo OIDC /oauth2/token por desincronización de reloj NTP",
            "tracker_id": 1,
            "status_id": 3,  # Resuelta
            "priority_id": 4,
            "description": """h2. Descripción del Problema
El backend de autenticación rechaza los tokens emitidos por Keycloak con el error:
```json
{"error": "invalid_grant", "error_description": "Token is not active yet (nbf claim in the future)"}
```

h2. Diagnóstico Técnico
El servidor de aplicaciones tenía un desfase de reloj local de +35 segundos respecto a la hora UTC real, haciendo que el claim @nbf@ (not before) del JWT emitido por Keycloak pareciera ser del futuro.

h2. Solución Aplicada (Documentada)
1. Se configuró y activó el servicio @systemd-timesyncd@ en los servidores backend apuntando a @pool.ntp.org@.
2. En la validación del JWT en Python (librería @PyJWT@), se agregó una tolerancia de reloj de 10 segundos:
```python
jwt.decode(token, key, algorithms=["RS256"], leeway=10)
```""",
            "notes": [
                "Solución probada y validada en producción. Los errores de invalid_grant desaparecieron por completo.",
            ],
        },
        {
            "project_id": "proyecto-prueba",
            "subject": "[Payments API] Duplicación de cargos en /api/v1/payments/charge por ausencia de encabezado Idempotency-Key",
            "tracker_id": 1,
            "status_id": 2,
            "priority_id": 5,  # Inmediata
            "description": """h2. Descripción del Problema
Se han detectado transacciones duplicadas para la misma orden de compra cuando los usuarios hacen doble clic en el botón 'Pagar ahora' o cuando la red móvil reintenta la petición tras un timeout prematuro.

h2. Endpoint y Comportamiento
@POST /api/v1/payments/charge@

h2. Evidencia
El cliente envía dos solicitudes con milisegundos de diferencia:
* Req 1: @10:04:12.100@ -> Cargo exitoso a tarjeta terminada en 4412 ($15.000).
* Req 2: @10:04:12.350@ -> Segundo cargo exitoso con distinta @transaction_id@ pero mismo @order_id@.

h2. Tarea Requerida
Implementar un middleware de idempotencia en FastAPI respaldado en Redis con TTL de 120 segundos utilizando el header @Idempotency-Key@.""",
            "notes": [
                "Se reembolsaron los cargos duplicados a los 14 clientes afectados.",
                "Middleware en desarrollo usando Redis SETNX.",
            ],
        },
        {
            "project_id": "proyecto-prueba",
            "subject": "[Facturación] Inconsistencia en redondeo de alícuota de IVA en notas de crédito con múltiples ítems",
            "tracker_id": 1,
            "status_id": 1,
            "priority_id": 3,
            "description": """h2. Síntoma
Al generar notas de crédito fiscales electrónicas con la API de AFIP/IRS, el importe total difiere en $0.01 o $0.02 respecto a la factura de origen cuando existen más de 5 ítems gravados con tasa del 21%.

h2. Causa
Se está aplicando @round(item_price * 0.21, 2)@ por ítem en lugar de sumar primero las bases imponibles y luego calcular el impuesto global.""",
            "notes": [],
        },
        {
            "project_id": "proyecto-prueba",
            "subject": "[Exportación] MemoryError al generar reportes XLSX con más de 100.000 registros en ReportingService",
            "tracker_id": 1,
            "status_id": 2,
            "priority_id": 3,
            "description": """h2. Síntomas
La biblioteca @openpyxl@ consume más de 2GB de RAM al intentar escribir hojas de cálculo con 120.000 filas y 45 columnas, provocando que el contenedor sea terminado.

h2. Propuesta
Migrar a @openpyxl(write_only=True)@ o generar archivos CSV comprimidos en zip cuando el número de filas supere los 20.000 registros.""",
            "notes": [
                "Se probó el modo write_only de openpyxl reduciendo el uso de memoria a menos de 180MB constantes.",
            ],
        },
        {
            "project_id": "proyecto-prueba",
            "subject": "[Frontend Web] Botón 'Confirmar Orden' no responde en navegadores Safari iOS 17.2 (requestSubmit API)",
            "tracker_id": 1,
            "status_id": 3,  # Resuelta
            "priority_id": 4,
            "description": """h2. Síntomas
Los usuarios de iPhone con iOS 17.2 y 17.3 reportan que al presionar el botón 'Confirmar Orden' en el modal de checkout, no ocurre ninguna acción y no se envía la petición de red.

h2. Causa
En @CheckoutModal.tsx@ se utilizaba @formRef.current.requestSubmit()@ dentro de un evento async sin polyfill. Safari lanza una excepción silenciosa si el elemento no está formalmente conectado al DOM al momento de la invocación.

h2. Solución Aplicada
Se implementó un fallback estándar usando @formRef.current.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }))@.""",
            "notes": [
                "Fix validado en BrowserStack en iPhone 14 / Safari 17.2. Desplegado en release v3.8.1.",
            ],
        },
        {
            "project_id": "proyecto-prueba",
            "subject": "[Users API] Error 422 Unprocessable Entity en validación de números telefónicos internacionales con prefijo '+'",
            "tracker_id": 1,
            "status_id": 1,
            "priority_id": 2,
            "description": r"""h2. Síntomas
Usuarios de España (+34) y México (+52) no pueden actualizar su perfil. La API devuelve:
```json
{"detail": [{"loc": ["body", "phone_number"], "msg": "value is not a valid phone number", "type": "value_error"}]}
```
La expresión regular actual @^\d{10}$@ solo admite teléfonos locales de 10 dígitos sin código de país.""",
            "notes": [],
        },
        {
            "project_id": "proyecto-prueba",
            "subject": "[Notificaciones] Error de CORS en endpoint Server-Sent Events /api/v1/notifications/stream",
            "tracker_id": 1,
            "status_id": 3,  # Resuelta
            "priority_id": 3,
            "description": """h2. Síntomas
La consola de Chrome muestra:
```log
Access to fetch at 'https://api.techvanguard.com/api/v1/notifications/stream' from origin 'https://admin.techvanguard.com' has been blocked by CORS policy: No 'Access-Control-Allow-Origin' header is present on the requested resource.
```

h2. Solución
Se incluyó @https://admin.techvanguard.com@ en la lista de @allow_origins@ del @CORSMiddleware@ de FastAPI y se habilitó @allow_credentials=True@ para solicitudes con EventSource.""",
            "notes": [
                "Verificado en entorno de staging. Conexión SSE fluye normalmente.",
            ],
        },
        {
            "project_id": "proyecto-prueba",
            "subject": "[WebSockets] Desconexión masiva de clientes con código 1008 (Policy Violation) al validar token JWT expirado",
            "tracker_id": 1,
            "status_id": 2,
            "priority_id": 3,
            "description": """h2. Síntomas
Cada 60 minutos (tiempo de vida del access token), miles de usuarios conectados a la sala de chat en vivo son desconectados simultáneamente en lugar de renovar la sesión silenciosamente mediante refresh token.

h2. Tarea
Permitir que el cliente envíe un mensaje @{"action": "refresh_auth", "token": "..."}@ antes de que el socket sea cerrado por el servidor.""",
            "notes": [
                "Diseño de protocolo de refresh en socket completado.",
            ],
        },
        {
            "project_id": "proyecto-prueba",
            "subject": "[Auth Service] Cabeceras de respuesta HTTP exponen JWT decodificado en trazas de error 403 Forbidden",
            "tracker_id": 1,
            "status_id": 3,  # Resuelta
            "priority_id": 5,  # Inmediata
            "description": """h2. Vulnerabilidad Detectada
Durante auditoría de seguridad se comprobó que al invocar un recurso sin permisos suficientes, la cabecera @X-Debug-Claims@ devolvía el payload completo del usuario (incluyendo roles internos y email corporativo).

h2. Remediación
Se eliminó la cabecera de depuración en entornos de producción y staging dentro del middleware @SecurityHeadersMiddleware@.""",
            "notes": [
                "Verificado con escaneo DAST con OWASP ZAP. Cabecera eliminada.",
            ],
        },
        {
            "project_id": "proyecto-prueba",
            "subject": "[Frontend Admin] Tabla de auditoría presenta solapamiento de columnas en monitores ultra-wide 21:9",
            "tracker_id": 1,
            "status_id": 1,
            "priority_id": 1,  # Baja
            "description": """h2. Descripción
En resoluciones 3440x1440px y 2560x1080px, las columnas 'Acción' y 'Usuario' se superponen debido a una regla de CSS @position: absolute@ mal aplicada en la clase @.audit-col-action@.""",
            "notes": [],
        },
        {
            "project_id": "proyecto-prueba",
            "subject": "[Integración Stripe] Webhook signature verification failed (HTTP 400) tras rotación de signing secret",
            "tracker_id": 1,
            "status_id": 2,
            "priority_id": 4,
            "description": """h2. Síntomas
Todas las llamadas de Stripe al endpoint @/api/v1/webhooks/stripe@ fallan con código 400.
```log
stripe.error.SignatureVerificationError: No signatures found matching the expected signature for payload
```

h2. Causa
Se rotó el endpoint signing secret en el dashboard de Stripe pero la variable de entorno @STRIPE_WEBHOOK_SECRET@ en el clúster de producción no fue actualizada.""",
            "notes": [
                "Se actualizó el secret en AWS Secrets Manager. Pendiente sincronización del ExternalSecret en Kubernetes.",
            ],
        },
        {
            "project_id": "proyecto-prueba",
            "subject": "[Catálogo SPA] Los parámetros de filtrado y búsqueda por facetas se pierden al navegar entre páginas",
            "tracker_id": 1,
            "status_id": 2,
            "priority_id": 2,
            "description": """h2. Pasos para Reproducir
1. Filtrar por categoría 'Laptops' y marca 'Dell'.
2. Hacer clic en la página 2 de resultados.
3. Observar que la URL cambia a @/catalog?page=2@ eliminando los query params @category@ y @brand@.""",
            "notes": [
                "Fix en revisión en PR #488 sincronizando el estado con useSearchParams() de React Router.",
            ],
        },
        {
            "project_id": "proyecto-prueba",
            "subject": "[Microservicios] Fallas esporádicas 401 Unauthorized en llamadas internas por expiración de certificado mTLS",
            "tracker_id": 1,
            "status_id": 1,
            "priority_id": 3,
            "description": """h2. Síntomas
La comunicación entre el servicio de inventario y el servicio de pagos falla con error 401 de forma intermitente durante 5 minutos cada día a las 00:00 UTC, coincidiendo con la rotación de certificados de Istio proxy.""",
            "notes": [],
        },
        {
            "project_id": "proyecto-prueba",
            "subject": "[S3 Reports] Peticiones POST /api/v1/reports quedan colgadas indefinidamente cuando el bucket S3 está en throttling",
            "tracker_id": 1,
            "status_id": 2,
            "priority_id": 3,
            "description": """h2. Descripción
Cuando AWS S3 devuelve una respuesta 503 SlowDown, el cliente de boto3 reintenta indefinidamente sin timeout de conexión fijado, manteniendo la conexión HTTP abierta con el cliente hasta que Nginx corta por 504 Gateway Timeout.""",
            "notes": [
                "Se añadirá una configuración de botocore.config.Config con connect_timeout=5 y read_timeout=15.",
            ],
        },
        {
            "project_id": "proyecto-prueba",
            "subject": "[Email Service] Caracteres especiales sin sanitizar provocan rotura visual en plantillas HTML de correo transaccional",
            "tracker_id": 1,
            "status_id": 3,  # Resuelta
            "priority_id": 2,
            "description": """h2. Descripción
Cuando un usuario se registra con caracteres como @&@, @<@ o @>@ en su nombre, la plantilla de Jinja2 renderizaba HTML malformado en el correo de bienvenida.

h2. Solución
Se activó el autoescape estricto en el Environment de Jinja2 (@autoescape=True@).""",
            "notes": [
                "Validado con tests unitarios que verifican el escape de entidades HTML.",
            ],
        },
        {
            "project_id": "proyecto-prueba",
            "subject": "[Carritos] Concurrencia en endpoint /api/v1/cart/items permite agregar cantidades negativas de producto",
            "tracker_id": 1,
            "status_id": 1,
            "priority_id": 4,
            "description": """h2. Vulnerabilidad de Lógica de Negocio
Al enviar múltiples peticiones concurrentes con @quantity: -1@ utilizando herramientas como Burp Suite o autocannon, un usuario puede reducir el precio total de su carrito por debajo de cero.""",
            "notes": [
                "Requiere validación pydantic con Field(gt=0) y transacción atómica en la base de datos.",
            ],
        },
        {
            "project_id": "proyecto-prueba",
            "subject": "[Mobile API] Respuestas comprimidas con Brotli provocan crash en clientes legacy de Android (versión < 10)",
            "tracker_id": 1,
            "status_id": 3,  # Resuelta
            "priority_id": 3,
            "description": """h2. Síntomas
La versión 1.4 de la app móvil en Android 9 se cierra inesperadamente al iniciar sesión. El logcat muestra @UnsatisfiedLinkError: libbrotli.so@.

h2. Solución
Se modificó Nginx para desactivar la compresión Brotli y usar exclusivamente Gzip si el User-Agent contiene @Android/9@ o @Android/8@.""",
            "notes": [
                "Validado en emulador de Android 9. Login operativo.",
            ],
        },
    ]


def populate_redmine(dry_run: bool = False, run_indexer: bool = False) -> None:
    """
    Puebla Redmine con el dataset de tickets técnicos realistas.
    Si dry_run es True, solo valida e imprime los tickets sin contactar a Redmine.
    """
    from src.redmine.client import RedmineClient

    tickets = get_realistic_tickets_dataset()
    print(f"Dataset cargado con {len(tickets)} tickets técnicos realistas.")

    if dry_run:
        print("\n--- MODO DRY-RUN ACTIVADO (No se crearán tickets en Redmine) ---")
        for i, t in enumerate(tickets, 1):
            print(f"[{i}/{len(tickets)}] Proyecto: {t['project_id']} | Asunto: {t['subject']}")
            print(
                f"   Tracker: {t['tracker_id']} | Prioridad: {t['priority_id']} | Estado: {t['status_id']}"
            )
            print(f"   Notas: {len(t.get('notes', []))} notas")
        print("\nValidación completada con éxito.")
        return

    try:
        with RedmineClient() as client:
            # Validar proyectos en Redmine
            available_projects = [p.get("identifier") for p in client.list_projects()]
            print(f"Proyectos detectados en Redmine: {available_projects}")

            created_count = 0
            for i, item in enumerate(tickets, 1):
                target_project = item["project_id"]
                # Fallback al primer proyecto disponible si el target no existe
                if target_project not in available_projects and available_projects:
                    print(
                        f"Aviso: Proyecto '{target_project}' no encontrado. Usando '{available_projects[0]}'"
                    )
                    target_project = available_projects[0]

                print(f"[{i}/{len(tickets)}] Creando: {item['subject']} en '{target_project}'...")
                created = client.create_issue(
                    project_id=target_project,
                    subject=item["subject"],
                    description=item["description"],
                    tracker_id=item["tracker_id"],
                    status_id=item["status_id"],
                    priority_id=item["priority_id"],
                )
                issue_id = created.get("id")
                print(f"   -> Creado Issue #{issue_id}")

                # Agregar notas / journals
                notes = item.get("notes", [])
                for note in notes:
                    print(f"      + Agregando nota técnica al Issue #{issue_id}...")
                    client.update_issue(issue_id=issue_id, notes=note)
                    time.sleep(0.3)

                created_count += 1

            print(
                f"\n¡Se crearon exitosamente {created_count} tickets con sus respectivos journals!"
            )

            if run_indexer:
                print("\nDisparando sincronización con Qdrant (RAGEngine)...")
                from src.jobs.indexer_job import run_sync

                run_sync()
                print("¡Indexación en Qdrant completada!")

    except Exception as e:
        print(f"Error al poblar Redmine: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generador de tickets de prueba técnicos y realistas."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simula la creación e imprime los tickets sin escribir en Redmine.",
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="Ejecuta la indexación en Qdrant tras crear los tickets.",
    )
    args = parser.parse_args()

    populate_redmine(dry_run=args.dry_run, run_indexer=args.index)
