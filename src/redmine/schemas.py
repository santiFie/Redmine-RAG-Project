"""
src/redmine/schemas.py
======================
Modelos Pydantic que representan los recursos de la API REST de Redmine.

Estos esquemas se usan para:
  - Validar y deserializar las respuestas JSON de Redmine
  - Tipar los argumentos de los nodos del grafo LangGraph
  - Facilitar la conversión a documentos de LlamaIndex

TODO:
  - Completar todos los campos opcionales según la documentación de Redmine
  - Agregar validadores custom (ej. fechas ISO 8601, URLs de attachments)
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class RedmineUser(BaseModel):
    """Representa un usuario referenciado en un issue."""

    id: int
    name: str


class RedmineIssueStatus(BaseModel):
    id: int
    name: str


class RedminePriority(BaseModel):
    id: int
    name: str


class RedmineProject(BaseModel):
    id: int
    identifier: str
    name: str
    description: str | None = None
    status: int | None = None
    created_on: datetime | None = None
    updated_on: datetime | None = None


class RedmineJournal(BaseModel):
    """Entrada de diario / nota de un issue (historial de cambios)."""

    id: int
    user: RedmineUser
    notes: str | None = None
    created_on: datetime
    details: list[dict] | None = None


class RedmineIssue(BaseModel):
    """
    Representación completa de un issue de Redmine.

    Campos relacionados (journals, attachments, relations) se obtienen
    con el parámetro ?include= en la API.
    """

    id: int
    project: RedmineProject | None = None
    subject: str
    description: str | None = None
    status: RedmineIssueStatus | None = None
    priority: RedminePriority | None = None
    author: RedmineUser | None = None
    assigned_to: RedmineUser | None = None
    created_on: datetime | None = None
    updated_on: datetime | None = None
    done_ratio: int = Field(default=0, ge=0, le=100)
    journals: list[RedmineJournal] = Field(default_factory=list)
    attachments: list[dict] | None = None
    relations: list[dict] | None = None
    watchers: list[dict] | None = None
    custom_fields: list[dict] | None = None

    def to_document_text(self) -> str:
        """Convierte el issue a texto plano para indexación con LlamaIndex."""
        issue_text = f"""
        ID: {self.id}
        Proyecto: {self.project.name if self.project else "N/A"}
        Asunto: {self.subject}
        Descripción: {self.description}
        Estado: {self.status.name if self.status else "N/A"}
        Prioridad: {self.priority.name if self.priority else "N/A"}
        Autor: {self.author.name if self.author else "N/A"}
        Asignado a: {self.assigned_to.name if self.assigned_to else "N/A"}
        Creado: {self.created_on}
        Actualizado: {self.updated_on}
        Porcentaje completado: {self.done_ratio}
        """

        return issue_text
