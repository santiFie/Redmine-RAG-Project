#!/usr/bin/env python3
"""
Ejemplo rápido para levantar Redmine (usando Docker) y crear un ticket de prueba
con el cliente RedmineClient.

Requisitos previos:
- Archivo .env en el proyecto con las variables:
    REDMINE_URL=https://localhost:3000  (o http://localhost:3000)
    REDMINE_API_KEY=your_api_key
- Activar el entorno virtual: source .venv/bin/activate
- Instalar dependencias (httpx, python-dotenv) en el venv.
"""

import os
import subprocess
import sys
from pathlib import Path

# Asegurarse de que estamos en el directorio del proyecto
# PROJECT_ROOT = Path(__file__).resolve().parents[2]
# os.chdir(PROJECT_ROOT)


def start_redmine():
    """Levanta Redmine usando docker‑compose (en modo detach)."""

    compose_file = "docker-compose.yml"
    if not Path(compose_file).exists():
        print(f"Archivo {compose_file} no encontrado. Asegúrate de estar en la raíz del proyecto.")
        sys.exit(1)

    print("Iniciando Redmine con docker‑compose...")
    subprocess.run(["docker", "compose", "up", "-d", "redmine"], check=True)
    print("Redmine iniciado. Espera unos segundos a que esté disponible.")


def create_test_issue():
    """Crea un ticket de prueba usando RedmineClient."""
    from client import RedmineClient

    project_id = os.getenv("REDMINE_TEST_PROJECT_ID", "my-project")
    issue = {
        "subject": "Ticket de prueba desde script",
        "description": "Este ticket se crea automáticamente para validar la integración.",
    }
    try:
        with RedmineClient() as client:
            created = client.create_issue(
                project_id=project_id,
                subject=issue["subject"],
                description=issue["description"],
            )
            print("Ticket creado exitosamente:")
            print(created)
    except Exception as e:
        print(f"Error al crear el ticket: {e}")


if __name__ == "__main__":
    # Paso 1: levantar Redmine (solo la primera vez, puedes comentar si ya está corriendo)
    # start_redmine()
    # Paso 2: crear el ticket de prueba
    create_test_issue()
