"""
mcp/redmine/server.py — MCP Server para Redmine usando FastMCP 3.x (stdio).

Herramientas (Pattern A — one-per-action):
  Issues  : get_issue, list_issues, create_issue, update_issue, delete_issue,
            add_watcher, remove_watcher
  Projects: list_projects, get_project, create_project, update_project, delete_project

Config via env vars: REDMINE_URL, REDMINE_API_KEY
"""

from __future__ import annotations

import os
from typing import Any

from client import (
    RedmineAPIError,
    RedmineClient,
    RedmineForbiddenError,
    RedmineNotFoundError,
    RedmineValidationError,
)
from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()

mcp = FastMCP(
    name="redmine",
    instructions=(
        "MCP server para gestionar issues y proyectos en Redmine vía REST API. "
        "Credenciales: REDMINE_URL y REDMINE_API_KEY."
    ),
)


def _get_client() -> RedmineClient:
    return RedmineClient(
        base_url=os.getenv("REDMINE_URL"),
        api_key=os.getenv("REDMINE_API_KEY"),
    )


def _fmt_err(e: Exception) -> str:
    if isinstance(e, RedmineNotFoundError):
        return f"Error 404: Recurso no encontrado. {e}"
    if isinstance(e, RedmineForbiddenError):
        return f"Error 403: Acceso denegado. {e}"
    if isinstance(e, RedmineValidationError):
        return f"Error 422: {'; '.join(e.errors)}"
    if isinstance(e, RedmineAPIError):
        return f"Error API Redmine: {e}"
    return f"Error inesperado: {e}"


# ── ISSUES ──────────────────────────────────────────────────────────────────


@mcp.tool
def get_issue(
    issue_id: int,
    include_journals: bool = False,
    include_attachments: bool = False,
    include_watchers: bool = False,
    include_relations: bool = False,
    include_children: bool = False,
) -> dict[str, Any]:
    """
    Obtiene los detalles completos de un issue por su ID.

    Args:
        issue_id: ID numérico del issue.
        include_journals: Incluir historial de cambios y comentarios.
        include_attachments: Incluir archivos adjuntos.
        include_watchers: Incluir lista de watchers.
        include_relations: Incluir relaciones con otros issues.
        include_children: Incluir sub-issues.
    """
    includes = [
        k
        for k, v in {
            "journals": include_journals,
            "attachments": include_attachments,
            "watchers": include_watchers,
            "relations": include_relations,
            "children": include_children,
        }.items()
        if v
    ]
    with _get_client() as c:
        try:
            return c.get_issue(issue_id, include=includes or None)
        except Exception as e:
            return {"error": _fmt_err(e)}


@mcp.tool
def list_issues(
    project_id: str | None = None,
    status_id: str = "open",
    assigned_to_id: str | None = None,
    tracker_id: int | None = None,
    sort: str | None = None,
    limit: int = 25,
    offset: int = 0,
    fetch_all: bool = False,
) -> list[dict[str, Any]]:
    """
    Lista issues de Redmine con filtros opcionales.

    Args:
        project_id: Identificador del proyecto. Omitir para todos los proyectos.
        status_id: "open" (default), "closed" o "*" para todos.
        assigned_to_id: ID del usuario asignado. Usar "me" para el autenticado.
        tracker_id: ID del tracker (Bug, Feature, etc.).
        sort: Columna de ordenamiento. Ej: "updated_on:desc".
        limit: Issues por página (máx. 100).
        offset: Desplazamiento para paginación.
        fetch_all: Si True, pagina automáticamente y devuelve todos.
    """
    with _get_client() as c:
        try:
            return c.list_issues(
                project_id=project_id,
                status_id=status_id,
                assigned_to_id=assigned_to_id,
                tracker_id=tracker_id,
                sort=sort,
                limit=limit,
                offset=offset,
                fetch_all=fetch_all,
            )
        except Exception as e:
            return [{"error": _fmt_err(e)}]


@mcp.tool
def create_issue(
    project_id: str,
    subject: str,
    description: str | None = None,
    tracker_id: int | None = None,
    status_id: int | None = None,
    priority_id: int | None = None,
    assigned_to_id: int | None = None,
    parent_issue_id: int | None = None,
    estimated_hours: float | None = None,
    is_private: bool = False,
) -> dict[str, Any]:
    """
    Crea un nuevo issue en un proyecto de Redmine.

    Args:
        project_id: Identificador del proyecto (requerido).
        subject: Título del issue (requerido).
        description: Descripción en texto plano o Textile.
        tracker_id: ID del tracker. Ej: 1=Bug, 2=Feature, 3=Support.
        status_id: ID del estado inicial.
        priority_id: ID de prioridad. Ej: 1=Baja, 2=Normal, 3=Alta, 4=Urgente.
        assigned_to_id: ID del usuario asignado.
        parent_issue_id: ID del issue padre para sub-issues.
        estimated_hours: Horas estimadas.
        is_private: Si True, solo visible para miembros con permisos.
    """
    with _get_client() as c:
        try:
            return c.create_issue(
                project_id=project_id,
                subject=subject,
                description=description,
                tracker_id=tracker_id,
                status_id=status_id,
                priority_id=priority_id,
                assigned_to_id=assigned_to_id,
                parent_issue_id=parent_issue_id,
                estimated_hours=estimated_hours,
                is_private=is_private or None,
            )
        except Exception as e:
            return {"error": _fmt_err(e)}


@mcp.tool
def update_issue(
    issue_id: int,
    subject: str | None = None,
    description: str | None = None,
    status_id: int | None = None,
    priority_id: int | None = None,
    assigned_to_id: int | None = None,
    tracker_id: int | None = None,
    estimated_hours: float | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """
    Actualiza un issue existente. Solo los campos provistos son modificados.

    Args:
        issue_id: ID del issue a modificar (requerido).
        subject: Nuevo título.
        description: Nueva descripción.
        status_id: ID del nuevo estado (para cerrar, reabrir, etc.).
        priority_id: ID de la nueva prioridad.
        assigned_to_id: ID del nuevo usuario asignado.
        tracker_id: ID del nuevo tracker.
        estimated_hours: Nuevas horas estimadas.
        notes: Comentario a añadir al historial del issue.
    """
    kwargs: dict[str, Any] = {
        k: v
        for k, v in dict(
            subject=subject,
            description=description,
            status_id=status_id,
            priority_id=priority_id,
            assigned_to_id=assigned_to_id,
            tracker_id=tracker_id,
            estimated_hours=estimated_hours,
        ).items()
        if v is not None
    }
    with _get_client() as c:
        try:
            success = c.update_issue(issue_id, notes=notes, **kwargs)
            return {"success": success, "issue_id": issue_id}
        except Exception as e:
            return {"error": _fmt_err(e)}


@mcp.tool
def delete_issue(issue_id: int) -> dict[str, Any]:
    """
    Elimina un issue permanentemente. ⚠️ Irreversible — confirmar con el usuario.

    Args:
        issue_id: ID del issue a eliminar.
    """
    with _get_client() as c:
        try:
            return {"success": c.delete_issue(issue_id), "issue_id": issue_id}
        except Exception as e:
            return {"error": _fmt_err(e)}


@mcp.tool
def add_watcher(issue_id: int, user_id: int) -> dict[str, Any]:
    """
    Añade un usuario como watcher (seguidor) de un issue.

    Args:
        issue_id: ID del issue.
        user_id: ID del usuario a añadir.
    """
    with _get_client() as c:
        try:
            return {"success": c.add_watcher(issue_id, user_id)}
        except Exception as e:
            return {"error": _fmt_err(e)}


@mcp.tool
def remove_watcher(issue_id: int, user_id: int) -> dict[str, Any]:
    """
    Elimina un usuario de los watchers de un issue.

    Args:
        issue_id: ID del issue.
        user_id: ID del usuario a eliminar.
    """
    with _get_client() as c:
        try:
            return {"success": c.remove_watcher(issue_id, user_id)}
        except Exception as e:
            return {"error": _fmt_err(e)}


# ── PROJECTS ─────────────────────────────────────────────────────────────────


@mcp.tool
def list_projects(
    limit: int = 25,
    offset: int = 0,
    fetch_all: bool = False,
) -> list[dict[str, Any]]:
    """
    Lista todos los proyectos accesibles con la API key configurada.

    Args:
        limit: Proyectos por página (máx. 100).
        offset: Desplazamiento para paginación.
        fetch_all: Si True, pagina automáticamente y devuelve todos.
    """
    with _get_client() as c:
        try:
            return c.list_projects(limit=limit, offset=offset, fetch_all=fetch_all)
        except Exception as e:
            return [{"error": _fmt_err(e)}]


@mcp.tool
def get_project(
    project_id: str,
    include_trackers: bool = False,
    include_issue_categories: bool = False,
) -> dict[str, Any]:
    """
    Obtiene los detalles de un proyecto por su ID o slug.

    Args:
        project_id: ID numérico o slug del proyecto. Ej: "mi-proyecto" o "42".
        include_trackers: Incluir trackers habilitados.
        include_issue_categories: Incluir categorías de issues.
    """
    includes = [
        k
        for k, v in {
            "trackers": include_trackers,
            "issue_categories": include_issue_categories,
        }.items()
        if v
    ]
    with _get_client() as c:
        try:
            return c.get_project(project_id, include=includes or None)
        except Exception as e:
            return {"error": _fmt_err(e)}


@mcp.tool
def create_project(
    name: str,
    identifier: str,
    description: str | None = None,
    is_public: bool = True,
    parent_id: int | None = None,
    inherit_members: bool = False,
) -> dict[str, Any]:
    """
    Crea un nuevo proyecto en Redmine.

    Args:
        name: Nombre completo del proyecto (requerido).
        identifier: Slug único para URLs, solo letras/números/guiones (requerido).
        description: Descripción del proyecto.
        is_public: Si True (default), el proyecto es público.
        parent_id: ID del proyecto padre para subproyectos.
        inherit_members: Si True, hereda miembros del padre.
    """
    with _get_client() as c:
        try:
            return c.create_project(
                name=name,
                identifier=identifier,
                description=description,
                is_public=is_public,
                parent_id=parent_id,
                inherit_members=inherit_members,
            )
        except Exception as e:
            return {"error": _fmt_err(e)}


@mcp.tool
def update_project(
    project_id: str,
    name: str | None = None,
    description: str | None = None,
    is_public: bool | None = None,
    inherit_members: bool | None = None,
) -> dict[str, Any]:
    """
    Actualiza los datos de un proyecto existente.

    Args:
        project_id: ID o slug del proyecto (requerido).
        name: Nuevo nombre.
        description: Nueva descripción.
        is_public: Cambiar visibilidad pública.
        inherit_members: Cambiar herencia de miembros.
    """
    kwargs = {
        k: v
        for k, v in dict(
            name=name,
            description=description,
            is_public=is_public,
            inherit_members=inherit_members,
        ).items()
        if v is not None
    }
    with _get_client() as c:
        try:
            success = c.update_project(project_id, **kwargs)
            return {"success": success, "project_id": project_id}
        except Exception as e:
            return {"error": _fmt_err(e)}


@mcp.tool
def delete_project(project_id: str) -> dict[str, Any]:
    """
    Elimina un proyecto y todos sus issues. ⚠️ Irreversible — confirmar con el usuario.

    Args:
        project_id: ID o slug del proyecto a eliminar.
    """
    with _get_client() as c:
        try:
            return {"success": c.delete_project(project_id), "project_id": project_id}
        except Exception as e:
            return {"error": _fmt_err(e)}


# ── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
