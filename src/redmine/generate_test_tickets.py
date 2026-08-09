#!/usr/bin/env python3
"""
Script para poblar Redmine con casos de prueba, comentarios y documentación simulada,
creando un escenario de una organización ficticia (TechVanguard Solutions) para
probar el RAGEngine de LlamaIndex.

Uso:
    python src/redmine/generate_test_tickets.py
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Asegurar que el path del proyecto esté en sys.path para importaciones
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.redmine.client import RedmineClient

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
                "Agregada nota sobre el bloqueo del puerto UDP en algunas redes de cafeterías."
            ]
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
                "Causa raíz identificada: El microservicio de reportes financieros filtraba conexiones. Se creó un ticket en desarrollo para parchear la fuga de memoria y conexiones."
            ]
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
                "Problema encontrado: El túnel IPSec tiene caídas intermitentes debido a una mala configuración de MTU en el router on-premise."
            ]
        },
        {
            "subject": "Error 502 Bad Gateway en el balanceador de carga Nginx",
            "description": """Los usuarios del sistema de inventario están experimentando errores esporádicos 502 Bad Gateway.
El problema ocurre principalmente durante las horas pico (10 AM a 2 PM).
El balanceador Nginx está configurado con upstream hacia tres nodos de la aplicación web (Node.js).""",
            "notes": [
                "He revisado los logs de Nginx y veo: `upstream prematurely closed connection while reading response header from upstream`.",
                "Parece que los nodos de Node.js se están reiniciando por falta de memoria (OOM Killer).",
                "He aumentado el límite de memoria de los contenedores Docker de 512MB a 1GB. Monitoreando."
            ]
        },
        {
            "subject": "[Wiki] Estándares de Seguridad de Contraseñas y MFA",
            "description": """h1. Estándares de Seguridad de TechVanguard

Todo empleado con acceso a sistemas de producción debe cumplir con los siguientes requisitos:

* *Contraseñas*: Mínimo 14 caracteres, alfanuméricas con símbolos. Debe cambiarse cada 90 días. No se pueden reutilizar las últimas 5 contraseñas.
* *MFA (Multi-Factor Authentication)*: Obligatorio para todos los sistemas (AWS, Redmine, VPN, Correo). Se recomiendan llaves FIDO2 o aplicaciones autenticadoras (Google Authenticator, Authy). No se permite el uso de SMS para MFA.

El equipo de IT auditará trimestralmente el cumplimiento de estas normas."""
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
                "Pendiente: Agregar paso de escaneo de imágenes de contenedor con Trivy antes del despliegue a producción."
            ]
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
                "Pendiente confirmación del líder técnico para otorgar credenciales de acceso a la DB de Staging."
            ]
        },
        {
            "subject": "Investigación: Fuga de Memoria (Memory Leak) en Servicio de Autenticación",
            "description": """Se ha observado un consumo incremental de memoria RAM en los Pods del servicio de autenticación (Keycloak 22.0).
El consumo pasa de 512MB a 3.8GB en un periodo de 48 horas hasta que el Pod es destruido por el OOMKilled de Kubernetes.

Se requiere un análisis del Heap Dump y métricas JVM.""",
            "notes": [
                "Tomamos un Heap Dump con `jcmd 1 GC.heap_dump /tmp/heap.hprof`. El analizador Eclipse MAT muestra que el 65% de los objetos retenidos corresponden a sesiones HTTP expiradas no liberadas.",
                "Se ajustó el Garbage Collector en las opciones JVM: `-XX:+UseG1GC -XX:MaxGCPauseMillis=200 -Xms1g -Xmx2g`.",
                "Parche aplicado: Se actualizó el parámetro `session-timeout` a 30 minutos y se habilitó la limpieza proactiva de sesiones inactivas en la base de datos de Redis."
            ]
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
            ]
        }
    ]

    try:
        with RedmineClient() as client:
            for item in issues_data:
                print(f"Creando ticket: {item['subject']}")
                created = client.create_issue(
                    project_id=project_id,
                    subject=item['subject'],
                    description=item['description'],
                )
                issue_id = created.get("id")
                
                print(f" -> Ticket #{issue_id} creado exitosamente.")
                
                # Agregar comentarios/notas (simulando journals/documentación en el tiempo)
                if "notes" in item:
                    for note in item["notes"]:
                        print(f"    Agregando nota al ticket #{issue_id}...")
                        client.update_issue(issue_id=issue_id, notes=note)
                        # Pequeña pausa para evitar colisiones o problemas de límite de tasa, si los hubiera
                        time.sleep(0.5)
                        
            print("\n¡Todos los datos de prueba han sido generados exitosamente!")
            
    except Exception as e:
        print(f"Error durante la generación de tickets: {e}")


if __name__ == "__main__":
    generate_test_data()
