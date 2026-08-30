#!/usr/bin/env python3
"""
Script para poblar Redmine con casos de prueba, comentarios y documentación simulada,
creando un escenario de una organización ficticia (TechVanguard Solutions) para
probar el RAGEngine de LlamaIndex.

Uso:
    python src/redmine/generate_test_tickets.py
"""

import os
import random
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Asegurar que el path del proyecto esté en sys.path para importaciones
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.redmine.client import RedmineClient  # noqa: E402

load_dotenv()


def generate_test_data():
    project_id = os.getenv("REDMINE_TEST_PROJECT_ID", "my-project")

    print(f"Generando datos de prueba en el proyecto: {project_id}...")

    issues_data = [
        {
            "subject": "[Documentación] Guía de Configuración de VPN (OpenVPN) para Nuevos Empleados",
            "description": """h1. Guía de Configuración VPN (TechVanguard)

Este documento sirve como base de conocimiento y wiki para la configuración de la VPN de la empresa.

h2. Pasos para la instalación
1. Descargar el cliente OpenVPN desde el portal interno (https://portal.techvanguard.local).
2. Solicitar el certificado de acceso a IT Soporte abriendo un ticket.
3. Importar el archivo @techvanguard_profile.ovpn@ en el cliente.
4. Conectar usando las credenciales de Active Directory.

h2. Solución de problemas frecuentes
* *Error TLS handshake failed*: Verifica que el reloj de tu sistema esté sincronizado.
* *Connection refused*: Asegúrate de que no estás en una red que bloquee el puerto UDP 1194.

Por favor, mantengan esta documentación actualizada.""",
            "notes": [
                "Actualizado el enlace del portal interno, ahora usamos https://vpn.techvanguard.com.",
                "Agregada nota sobre el bloqueo del puerto UDP en algunas redes de cafeterías.",
            ],
        },
        {
            "subject": "Incidencia Crítica: Caída de la base de datos de producción (PostgreSQL)",
            "description": """Se reporta que la base de datos principal de producción de PostgreSQL (v14) dejó de aceptar conexiones a las 03:00 AM. 
Los logs muestran el siguiente error:
`FATAL: sorry, too many clients already`

Es necesario investigar la causa de la saturación del pool de conexiones.""",
            "notes": [
                "Revisando los logs de la aplicación, parece que un cronjob de reportes masivos no está cerrando las conexiones correctamente.",
                "Solución temporal aplicada: Se reinició el servicio de pgBouncer y se incrementó max_connections a 500 en postgresql.conf. El servicio está restablecido.",
                "Causa raíz identificada: El microservicio de reportes financieros filtraba conexiones. Se creó un ticket en desarrollo para parchear la fuga de memoria y conexiones.",
            ],
        },
        {
            "subject": "[Infra] Migración de servidores On-Premise a AWS EC2",
            "description": """Este ticket consolida el plan de migración de nuestros servidores físicos alojados en la oficina central hacia AWS.

*Fase 1: Assessment*
- Mapeo de dependencias de red.
- Evaluación de costos de instancias EC2 y volúmenes EBS.

*Fase 2: Ejecución*
- Setup de la VPC en AWS.
- Configuración de túnel IPSec Site-to-Site.
- Sincronización de datos con AWS DataSync.

Adjuntaremos los avances como comentarios.""",
            "notes": [
                "Fase 1 completada. El costo estimado mensual es de $1,200 USD. Procedemos con la Fase 2.",
                "VPC creada con las subredes 10.0.1.0/24 (Pública) y 10.0.2.0/24 (Privada).",
                "Problema encontrado: El túnel IPSec tiene caídas intermitentes debido a una mala configuración de MTU en el router on-premise.",
            ],
        },
        {
            "subject": "Error 502 Bad Gateway en el balanceador de carga Nginx",
            "description": """Los usuarios del sistema de inventario están experimentando errores esporádicos 502 Bad Gateway.
El problema ocurre principalmente durante las horas pico (10 AM a 2 PM).
El balanceador Nginx está configurado con upstream hacia tres nodos de la aplicación web (Node.js).""",
            "notes": [
                "He revisado los logs de Nginx y veo: `upstream prematurely closed connection while reading response header from upstream`.",
                "Parece que los nodos de Node.js se están reiniciando por falta de memoria (OOM Killer).",
                "He aumentado el límite de memoria de los contenedores Docker de 512MB a 1GB. Monitoreando.",
            ],
        },
        {
            "subject": "[Wiki] Estándares de Seguridad de Contraseñas y MFA",
            "description": """h1. Estándares de Seguridad de TechVanguard

Todo empleado con acceso a sistemas de producción debe cumplir con los siguientes requisitos:

* *Contraseñas*: Mínimo 14 caracteres, alfanuméricas con símbolos. Debe cambiarse cada 90 días. No se pueden reutilizar las últimas 5 contraseñas.
* *MFA (Multi-Factor Authentication)*: Obligatorio para todos los sistemas (AWS, Redmine, VPN, Correo). Se recomiendan llaves FIDO2 o aplicaciones autenticadoras (Google Authenticator, Authy). No se permite el uso de SMS para MFA.

El equipo de IT auditará trimestralmente el cumplimiento de estas normas.""",
        },
        {
            "subject": "[DevOps] Implementación de Pipeline CI/CD con GitHub Actions y Kubernetes (EKS)",
            "description": """h1. Pipeline CI/CD para Microservicios

Estamos automatizando el despliegue del microservicio de pagos en el clúster EKS de AWS usando GitHub Actions.

h2. Flujo del Pipeline
1. *Build & Test*: Se ejecutan pruebas unitarias y linters en cada Pull Request.
2. *Docker Build*: Al mergear a @main@, se construye la imagen Docker y se sube a AWS ECR (@1234567890.dkr.ecr.us-east-1.amazonaws.com/techvanguard/payments@).
3. *Deploy*: Se actualiza el manifiesto Helm en el clúster de Kubernetes en el namespace @production@.

h2. Variables y Secretos Requeridos
- @AWS_ACCESS_KEY_ID@
- @AWS_SECRET_ACCESS_KEY@
- @KUBE_CONFIG_DATA@""",
            "notes": [
                "El primer despliegue falló por falta de permisos en el rol IAM `GitHubActionsEKSDeployerRole`.",
                "Se adjuntó la política `AmazonEKSClusterPolicy` al rol y el despliegue en staging concluyó exitosamente.",
                "Pendiente: Agregar paso de escaneo de imágenes de contenedor con Trivy antes del despliegue a producción.",
            ],
        },
        {
            "subject": "[Soporte IT] Solicitud de permisos y accesos para nueva desarrolladora Backend",
            "description": """Solicitud de onboarding para la nueva desarrolladora Backend: **Ana Martínez** (ana.martinez@techvanguard.com).

Se requiere acceso a los siguientes recursos:
- [x] Cuenta de Google Workspace / Email Corporativo.
- [ ] Usuario IAM en AWS con políticas de lectura en S3 y CloudWatch.
- [ ] Acceso de lectura a la base de datos PostgreSQL de Staging.
- [ ] Acceso a los repositorios de GitHub en la organización TechVanguardDev.""",
            "notes": [
                "Cuenta de email y GitHub activados. Agregada al equipo `Backend-Team` en GitHub.",
                "Creado usuario IAM `ana.martinez` con autenticación MFA obligatoria.",
                "Pendiente confirmación del líder técnico para otorgar credenciales de acceso a la DB de Staging.",
            ],
        },
        {
            "subject": "Investigación: Fuga de Memoria (Memory Leak) en Servicio de Autenticación",
            "description": """Se ha observado un consumo incremental de memoria RAM en los Pods del servicio de autenticación (Keycloak 22.0).
El consumo pasa de 512MB a 3.8GB en un periodo de 48 horas hasta que el Pod es destruido por el OOMKilled de Kubernetes.

Se requiere un análisis del Heap Dump y métricas JVM.""",
            "notes": [
                "Tomamos un Heap Dump con `jcmd 1 GC.heap_dump /tmp/heap.hprof`. El analizador Eclipse MAT muestra que el 65% de los objetos retenidos corresponden a sesiones HTTP expiradas no liberadas.",
                "Se ajustó el Garbage Collector en las opciones JVM: `-XX:+UseG1GC -XX:MaxGCPauseMillis=200 -Xms1g -Xmx2g`.",
                "Parche aplicado: Se actualizó el parámetro `session-timeout` a 30 minutos y se habilitó la limpieza proactiva de sesiones inactivas en la base de datos de Redis.",
            ],
        },
        {
            "subject": "[Wiki] Plan de Recuperación ante Desastres (Disaster Recovery & Backup Policy)",
            "description": """h1. Plan de Disaster Recovery (DRP) - TechVanguard

h2. Objetivos de Recuperación
* *RPO (Recovery Point Objective)*: Máximo 1 hora de pérdida de datos.
* *RTO (Recovery Time Objective)*: Recuperación total de servicios críticos en menos de 4 horas.

h2. Estrategia de Copias de Seguridad
1. *Bases de Datos PostgreSQL*: Backups automatizados cada hora guardados en S3 con Lifecycle Policy (retención 30 días, luego Glacier).
2. *Archivos de Configuración y Código*: Control de versiones en GitHub y backups diarios de volúmenes EBS en AWS.

h2. Procedimiento de Restauración de Base de Datos
```bash
aws s3 cp s3://techvanguard-backups-postgres/latest.dump /tmp/latest.dump
pg_restore --clean --if-exists -h localhost -U postgres -d techvanguard_prod /tmp/latest.dump
```""",
            "notes": [
                "Simulacro de restauración realizado el 15 de julio: Tiempo de recuperación efectivo RTO fue de 2 horas y 15 minutos. Cumple con los SLAs exigidos."
            ],
        },
    ]

    # Parámetros estándar para tickets aleatorios
    trackers = [1, 2, 3]  # 1: Error, 2: Tarea, 3: Soporte
    statuses = [
        1,
        2,
        3,
        4,
        5,
        6,
    ]  # 1: Nueva, 2: En curso, 3: Resuelta, 4: Comentarios, 5: Cerrada, 6: Rechazada
    priorities = [1, 2, 3, 4, 5]  # 1: Baja, 2: Normal, 3: Alta, 4: Urgente, 5: Inmediata

    realistic_templates = [
        {
            "subject": "Alerta de Seguridad: Vulnerabilidad Crítica (CVE-{cve_id}) en {component}",
            "description": "h1. Reporte de Vulnerabilidad en {component}\n\nEl escáner de seguridad ha detectado la vulnerabilidad *CVE-{cve_id}*.\n\nh2. Detalles Técnicos\n* *Severidad*: Crítica (CVSS 9.8)\n* *Descripción*: Un atacante podría ejecutar código remoto si envía un payload malicioso modificado.\n\nh2. Plan de Remediación\n1. Actualizar la imagen base del contenedor de `{component}` a la versión con el parche aplicado.\n2. Ejecutar escaneo con Trivy.\n3. Desplegar en Staging y ejecutar pruebas de regresión.\n\n```yaml\n# Ejemplo de parche en Dockerfile\n- FROM alpine:3.18\n+ FROM alpine:3.19\n```\n\nPor favor, asignar prioridad inmediata.",
            "notes": [
                "Se ha creado un branch para aplicar el parche de seguridad.",
                "El parche ha pasado las pruebas en entorno de QA. Procedemos con el paso a producción.",
                "Despliegue completado. El nuevo escaneo de Trivy muestra 0 vulnerabilidades críticas.",
            ],
        },
        {
            "subject": "Problema de Rendimiento: Consultas lentas en {component}",
            "description": "h1. Degradación de Rendimiento\n\nEl sistema de monitoreo Datadog ha detectado un aumento significativo en la latencia del P99 para el `{component}`.\n\nh2. Evidencia\nLas consultas tardan más de 5 segundos en promedio. Query identificada en el slow query log:\n\n```sql\nSELECT * FROM transacciones WHERE user_id = {random_id} AND status = 'PENDING' ORDER BY created_at DESC;\n```\n\nh2. Posible causa\nFalta un índice compuesto en las columnas `(user_id, status, created_at)`.\n\nRevisar el plan de ejecución (`EXPLAIN ANALYZE`) y proponer la migración.",
            "notes": [
                "He revisado el EXPLAIN ANALYZE y efectivamente se está haciendo un Seq Scan en la tabla que tiene más de 10 millones de registros.",
                "Se ha generado el script de migración Flyway para agregar el índice `CONCURRENTLY`.",
                "Migración ejecutada en producción. La latencia bajó de 5s a 45ms. Problema resuelto.",
            ],
        },
        {
            "subject": "Incidencia: Pods de {component} reiniciando constantemente (CrashLoopBackOff)",
            "description": "h1. Alerta de Kubernetes\n\nEl namespace `production` está reportando un estado `CrashLoopBackOff` para los pods del despliegue `{component}`.\n\nh2. Logs del Contenedor\n```log\n[ERROR] 2023-10-25 08:15:32 - FATAL: Connection to Redis failed at redis-cluster.internal:6379\n[ERROR] 2023-10-25 08:15:32 - TimeoutError: connect ETIMEDOUT 10.0.{subnet}.45:6379\n```\n\nh2. Impacto\nLos usuarios no pueden iniciar sesión porque el sistema de caché y sesiones está inaccesible desde este microservicio.\nRevisar las reglas de NetworkPolicy o el estado del cluster de Redis.",
            "notes": [
                "Se verificó que Redis está operativo, pero hubo un cambio reciente en las NetworkPolicies de Calico.",
                "El cambio bloqueó el tráfico saliente en el puerto 6379 desde el namespace del componente. Se está revirtiendo el commit.",
                "Commit revertido y pods en estado Running nuevamente.",
            ],
        },
        {
            "subject": "[Wiki] Procedimiento de Onboarding Técnico para {component}",
            "description": "h1. Guía de Inicio Rápido: {component}\n\nBienvenidos al repositorio principal de `{component}`.\n\nh2. Requisitos Previos\n* Docker y Docker Compose v2\n* Python 3.11+\n* Node.js 20.x (para los assets estáticos)\n\nh2. Instalación Local\n1. Clonar el repositorio y configurar variables de entorno:\n```bash\ncp .env.example .env\n# Solicitar la clave de API de desarrollo a un administrador\n```\n2. Levantar la base de datos de desarrollo:\n```bash\ndocker-compose up -d db redis\n```\n3. Ejecutar las migraciones:\n```bash\nmake migrate\n```\n\nh2. Arquitectura\nEste servicio se comunica mediante gRPC con el backend central y publica eventos en Kafka (topic: `events.{component}`).",
            "notes": [
                "Se actualizó la guía para incluir la dependencia de Node.js 20.x, ya que antes usábamos la 18.",
                "Añadida la aclaración sobre cómo solicitar las credenciales de desarrollo en Vault.",
            ],
        },
        {
            "subject": "Error 504 Gateway Timeout en la API de {component}",
            "description": "h1. Reporte de Error en Producción\n\nDurante el pico de tráfico de las 14:00, el balanceador de carga (AWS ALB) devolvió múltiples errores `504 Gateway Timeout` hacia `{component}`.\n\nh2. Métricas Observadas\n* *CPU*: 99% en todos los nodos del Auto Scaling Group.\n* *Latencia Promedio*: > 30 segundos.\n* *Conexiones Activas*: 5000+ (límite configurado en uwsgi es 2000).\n\nh2. Pasos a seguir\n1. Escalar horizontalmente de forma manual (añadir 3 nodos extra).\n2. Investigar qué endpoint está consumiendo tantos recursos.\n3. Ajustar los umbrales del Target Tracking Scaling Policy.",
            "notes": [
                "El escalado manual mitigó el problema. El tráfico se ha estabilizado.",
                "Descubrimos que el endpoint `/api/v1/export` estaba siendo llamado recursivamente por un script malicioso.",
                "Se ha implementado rate limiting en el WAF para la ruta de exportación. Monitoreando.",
            ],
        },
        {
            "subject": "Renovación de Certificados Let's Encrypt en {component}",
            "description": "h1. Certificado SSL próximo a expirar\n\nEl sistema de monitoreo indica que el certificado SSL para el dominio asociado a `{component}` expira en 5 días.\n\nh2. Detalles\n* *Dominio*: `api.{component}.techvanguard.com`\n* *Emisor*: Let's Encrypt\n* *Servidor*: Nginx en instancia EC2 (IP: 10.0.{subnet}.{random_id})\n\nh2. Acción Requerida\nNormalmente el certbot debería renovar automáticamente, pero parece que el cronjob falló. Revisar los logs de certbot en `/var/log/letsencrypt/letsencrypt.log` y ejecutar la renovación manual si es necesario:\n\n```bash\nsudo certbot renew --force-renewal\nsudo systemctl reload nginx\n```",
            "notes": [
                "Revisando los logs, certbot falló por un problema de resolución DNS temporal durante la validación HTTP-01.",
                "He ejecutado la renovación manual exitosamente y recargado Nginx.",
                "Se configuró una alerta adicional en Datadog para monitorear los fallos del cronjob de certbot en el futuro.",
            ],
        },
        {
            "subject": "[DevOps] Migración de CI/CD hacia GitLab CI para {component}",
            "description": "h1. Plan de Migración de Pipeline\n\nComo parte de la estandarización de herramientas, necesitamos mover los pipelines de Jenkins hacia GitLab CI para `{component}`.\n\nh2. Etapas del Pipeline a Migrar\n1. *Linting*: Flake8 y Black para código Python.\n2. *Testing*: Pytest con cobertura mínima del 85%.\n3. *Build*: Construcción de imagen Docker y push al Registry interno.\n4. *Deploy*: Despliegue usando ArgoCD.\n\nh2. Ejemplo del archivo .gitlab-ci.yml inicial propuesto\n```yaml\nstages:\n  - test\n  - build\n\nrun_tests:\n  stage: test\n  image: python:3.11-slim\n  script:\n    - pip install -r requirements.txt\n    - pytest --cov=src/\n```\n\nAsignar a un ingeniero DevOps para completar la configuración de ArgoCD.",
            "notes": [
                "Pipeline básico implementado en GitLab. Falta integrar la autenticación con el Registry.",
                "Se configuraron las variables CI/CD protegidas. El paso de build ya funciona.",
                "Migración completada. ArgoCD ha sincronizado exitosamente la primera imagen generada por GitLab CI. Apagando el job en Jenkins.",
            ],
        },
        {
            "subject": "Bug Visual: Error de alineación en Safari para {component}",
            "description": "h1. Reporte de Bug de UI\n\nLos usuarios de MacOS/iOS reportan que en el navegador Safari, los botones de acción del `{component}` están superpuestos con el texto principal.\n\nh2. Pasos para reproducir\n1. Abrir Safari (versión 16.0+).\n2. Navegar a la pantalla principal del `{component}`.\n3. Observar la barra de herramientas inferior.\n\nh2. Contexto Técnico\nParece estar relacionado con el uso de `flex-gap` y un fallback faltante en el CSS compilado. \n\n```css\n.toolbar {\n  display: flex;\n  gap: 16px; /* Safari antiguo tiene bugs con gap en flexbox */\n  justify-content: space-between;\n}\n```\n\nAplicar vendor prefixes o reescribir el layout usando márgenes según sea necesario.",
            "notes": [
                "Confirmado. El problema ocurre en Safari 14.1 específicamente. Añadiendo fix.",
                "El PR #442 resuelve el problema implementando márgenes en lugar de gap condicionalmente para navegadores legacy.",
                "Desplegado a producción. Bug cerrado.",
            ],
        },
    ]

    components_list = [
        "Auth Service",
        "Payment Gateway",
        "User Dashboard",
        "Inventory API",
        "Notification Worker",
        "Data Pipeline",
        "CRM Integration",
        "Mobile App Backend",
        "Frontend SPA",
        "Admin Panel",
    ]

    try:
        with RedmineClient() as client:
            # Detección dinámica de proyectos y usuarios disponibles en Redmine
            detected_projects = [p["identifier"] for p in client.list_projects() if p.get("identifier")]
            projects = detected_projects if detected_projects else [project_id]

            detected_users = [u["id"] for u in client.list_users() if u.get("id")]
            users = detected_users if detected_users else [1]

            print(f"Proyectos detectados ({len(projects)}): {projects}")
            print(f"Usuarios detectados ({len(users)}): {users}")

            print("Generando 50 tickets aleatorios adicionales con ejemplos reales...")
            for _ in range(50):
                template = random.choice(realistic_templates)
                comp = random.choice(components_list)
                cve_id = f"2023-{random.randint(1000, 99999)}"
                subnet = random.randint(10, 200)
                rnd_id = random.randint(100, 9999)

                subj = template["subject"].replace("{component}", comp).replace("{cve_id}", cve_id)
                desc = (
                    template["description"]
                    .replace("{component}", comp)
                    .replace("{cve_id}", cve_id)
                    .replace("{subnet}", str(subnet))
                    .replace("{random_id}", str(rnd_id))
                )

                max_notes = len(template["notes"])
                num_notes = random.randint(0, max_notes)
                issue_notes = template["notes"][:num_notes] if num_notes > 0 else []

                issues_data.append(
                    {
                        "subject": subj,
                        "description": desc,
                        "notes": issue_notes,
                        "tracker_id": random.choice(trackers),
                        "status_id": random.choice(statuses),
                        "priority_id": random.choice(priorities),
                        "assigned_to_id": random.choice(users),
                        "project_id": random.choice(projects),
                    }
                )

            for item in issues_data:
                t_project_id = item.get("project_id", project_id)
                print(f"Creando ticket: {item['subject']} en proyecto {t_project_id}")

                kwargs = {}
                if "tracker_id" in item:
                    kwargs["tracker_id"] = item["tracker_id"]
                if "status_id" in item:
                    kwargs["status_id"] = item["status_id"]
                if "priority_id" in item:
                    kwargs["priority_id"] = item["priority_id"]
                if "assigned_to_id" in item:
                    kwargs["assigned_to_id"] = item["assigned_to_id"]

                created = client.create_issue(
                    project_id=t_project_id,
                    subject=item["subject"],
                    description=item["description"],
                    **kwargs,
                )
                issue_id = created.get("id")

                print(f" -> Ticket #{issue_id} creado exitosamente.")

                if "notes" in item and item["notes"]:
                    for note in item["notes"]:
                        print(f"    Agregando nota al ticket #{issue_id}...")
                        client.update_issue(issue_id=issue_id, notes=note)
                        time.sleep(0.5)

            print("\n¡Todos los datos de prueba han sido generados exitosamente!")

    except Exception as e:
        print(f"Error durante la generación de tickets: {e}")


if __name__ == "__main__":
    generate_test_data()
