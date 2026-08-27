"""
src/redmine/client.py
=====================
Cliente base para la API REST de Redmine.

Responsabilidades:
  - Autenticación mediante API Key (header X-Redmine-API-Key)
  - Operaciones CRUD genéricas sobre cualquier recurso de Redmine
  - Manejo de errores HTTP tipados
  - Soporte para paginación (offset / limit)

Recursos soportados:
  - Issues      : /issues.json
  - Projects    : /projects.json
  - Journals    : via /issues/<id>.json?include=journals
  - Watchers    : /issues/<id>/watchers.json
  - Attachments : /attachments/<id>.json

Uso previsto:
    client = RedmineClient()
    issue = client.get_issue(issue_id=42)
    issues = client.list_issues(project_id="my-project", status_id="open")
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Excepciones tipadas
# ---------------------------------------------------------------------------


class RedmineAPIError(Exception):
    """Error genérico de la API de Redmine."""


class RedmineNotFoundError(RedmineAPIError):
    """Recurso no encontrado (HTTP 404)."""


class RedmineForbiddenError(RedmineAPIError):
    """Acceso denegado (HTTP 403)."""


class RedmineValidationError(RedmineAPIError):
    """Error de validación (HTTP 422). Incluye los mensajes del servidor."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(f"Validation errors: {'; '.join(errors)}")


# ---------------------------------------------------------------------------
# Cliente principal
# ---------------------------------------------------------------------------


class RedmineClient:
    """
    Cliente HTTP síncrono para la API REST de Redmine.

    Autenticación via X-Redmine-API-Key header.
    Documentación oficial: https://www.redmine.org/projects/redmine/wiki/Rest_api
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("REDMINE_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("REDMINE_API_KEY", "")

        if not self.base_url:
            raise ValueError("REDMINE_URL no configurada. Verificar .env")
        if not self.api_key:
            raise ValueError("REDMINE_API_KEY no configurada. Verificar .env")

        self._headers = {
            "X-Redmine-API-Key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self._timeout = timeout
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=self._headers,
            timeout=self._timeout,
        )

    def close(self) -> None:
        """Cierra el cliente HTTP subyacente."""
        self._client.close()

    def __enter__(self) -> RedmineClient:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Issues  — https://www.redmine.org/projects/redmine/wiki/Rest_Issues
    # ------------------------------------------------------------------

    def get_issue(self, issue_id: int, include: list[str] | None = None) -> dict[str, Any]:
        """
        Obtiene un issue por su ID.

        GET /issues/{issue_id}.json

        Args:
            issue_id: ID del issue en Redmine.
            include: Relaciones a incluir. Valores posibles:
                     "children", "attachments", "relations", "changesets",
                     "journals", "watchers", "allowed_statuses".

        Returns:
            Diccionario con los datos del issue (campo ``issue``).

        Raises:
            RedmineNotFoundError: Si el issue no existe (404).
            RedmineForbiddenError: Si no se tiene acceso (403).
            RedmineAPIError: Para cualquier otro error HTTP o de red.
        """
        params: dict[str, Any] = {}
        if include:
            params["include"] = ",".join(include)

        data = self._get(f"/issues/{issue_id}.json", params=params)
        return data["issue"]

    def list_issues(
        self,
        project_id: str | int | None = None,
        status_id: str = "open",
        assigned_to_id: int | str | None = None,
        tracker_id: int | None = None,
        subproject_id: str | int | None = None,
        parent_id: int | None = None,
        issue_id: str | int | None = None,
        sort: str | None = None,
        include: list[str] | None = None,
        limit: int = 25,
        offset: int = 0,
        fetch_all: bool = False,
        **extra_filters: Any,
    ) -> list[dict[str, Any]]:
        """
        Lista issues con filtros opcionales.

        GET /issues.json

        Args:
            project_id: Identificador (string) o ID (int) del proyecto.
            status_id: Estado del issue. Valores: "open" (por defecto),
                       "closed" o "*" para todos. También acepta un ID numérico.
            assigned_to_id: ID del usuario asignado. Use "me" para el usuario
                            autenticado actualmente.
            tracker_id: ID del tracker.
            subproject_id: ID del subproyecto. Use "!*" para excluirlos.
            parent_id: ID del issue padre.
            issue_id: Un ID o lista de IDs separados por coma.
            sort: Columna de ordenamiento, p.ej. "updated_on:desc".
            include: Asociaciones a incluir: "attachments", "relations".
            limit: Máx. issues por página (max 100 en Redmine).
            offset: Desplazamiento para paginación.
            fetch_all: Si es True, itera automáticamente sobre todas las
                       páginas y devuelve todos los issues.
            **extra_filters: Filtros adicionales (ej. created_on, updated_on,
                             cf_1="valor", etc.).

        Returns:
            Lista de diccionarios con los issues.
        """
        params: dict[str, Any] = {
            "status_id": status_id,
            "limit": limit,
            "offset": offset,
        }
        if project_id is not None:
            params["project_id"] = project_id
        if assigned_to_id is not None:
            params["assigned_to_id"] = assigned_to_id
        if tracker_id is not None:
            params["tracker_id"] = tracker_id
        if subproject_id is not None:
            params["subproject_id"] = subproject_id
        if parent_id is not None:
            params["parent_id"] = parent_id
        if issue_id is not None:
            params["issue_id"] = issue_id
        if sort is not None:
            params["sort"] = sort
        if include:
            params["include"] = ",".join(include)
        params.update(extra_filters)

        if not fetch_all:
            data = self._get("/issues.json", params=params)
            return data.get("issues", [])

        # Paginación automática
        all_issues: list[dict[str, Any]] = []
        current_offset = offset
        while True:
            params["offset"] = current_offset
            data = self._get("/issues.json", params=params)
            page = data.get("issues", [])
            all_issues.extend(page)
            total_count: int = data.get("total_count", 0)
            current_offset += len(page)
            if current_offset >= total_count or not page:
                break
        return all_issues

    def create_issue(
        self,
        project_id: str | int,
        subject: str,
        description: str | None = None,
        tracker_id: int | None = None,
        status_id: int | None = None,
        priority_id: int | None = None,
        assigned_to_id: int | None = None,
        category_id: int | None = None,
        fixed_version_id: int | None = None,
        parent_issue_id: int | None = None,
        watcher_user_ids: list[int] | None = None,
        is_private: bool | None = None,
        estimated_hours: float | None = None,
        custom_fields: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Crea un nuevo issue en Redmine.

        POST /issues.json

        Args:
            project_id: Identificador del proyecto destino (requerido).
            subject: Título del issue (requerido).
            description: Descripción detallada.
            tracker_id: ID del tracker (Bug, Feature, etc.).
            status_id: ID del estado inicial.
            priority_id: ID de la prioridad.
            assigned_to_id: ID del usuario asignado.
            category_id: ID de la categoría.
            fixed_version_id: ID de la versión objetivo.
            parent_issue_id: ID del issue padre.
            watcher_user_ids: Lista de IDs de usuarios a añadir como watchers.
            is_private: True si el issue debe ser privado.
            estimated_hours: Horas estimadas.
            custom_fields: Lista de dicts {"id": int, "value": ...}.
            **kwargs: Campos adicionales a incluir en el payload.

        Returns:
            Diccionario con el issue recién creado (campo ``issue``).

        Raises:
            RedmineValidationError: Si el servidor devuelve errores de validación (422).
            RedmineAPIError: Para cualquier otro error HTTP o de red.
        """
        issue_payload: dict[str, Any] = {
            "project_id": project_id,
            "subject": subject,
        }
        if description is not None:
            issue_payload["description"] = description
        if tracker_id is not None:
            issue_payload["tracker_id"] = tracker_id
        if status_id is not None:
            issue_payload["status_id"] = status_id
        if priority_id is not None:
            issue_payload["priority_id"] = priority_id
        if assigned_to_id is not None:
            issue_payload["assigned_to_id"] = assigned_to_id
        if category_id is not None:
            issue_payload["category_id"] = category_id
        if fixed_version_id is not None:
            issue_payload["fixed_version_id"] = fixed_version_id
        if parent_issue_id is not None:
            issue_payload["parent_issue_id"] = parent_issue_id
        if watcher_user_ids is not None:
            issue_payload["watcher_user_ids"] = watcher_user_ids
        if is_private is not None:
            issue_payload["is_private"] = is_private
        if estimated_hours is not None:
            issue_payload["estimated_hours"] = estimated_hours
        if custom_fields is not None:
            issue_payload["custom_fields"] = custom_fields
        issue_payload.update(kwargs)

        data = self._post("/issues.json", payload={"issue": issue_payload})
        return data["issue"]

    def update_issue(self, issue_id: int, notes: str | None = None, **kwargs: Any) -> bool:
        """
        Actualiza un issue existente.

        PUT /issues/{issue_id}.json

        Args:
            issue_id: ID del issue a modificar.
            notes: Comentario a añadir como nota del journal.
            **kwargs: Campos a actualizar (status_id, priority_id,
                      subject, assigned_to_id, custom_fields, etc.).

        Returns:
            True si la actualización fue exitosa (HTTP 204).

        Raises:
            RedmineNotFoundError: Si el issue no existe (404).
            RedmineValidationError: Si los datos son inválidos (422).
            RedmineAPIError: Para cualquier otro error HTTP o de red.
        """
        issue_payload: dict[str, Any] = {**kwargs}
        if notes is not None:
            issue_payload["notes"] = notes

        return self._put(f"/issues/{issue_id}.json", payload={"issue": issue_payload})

    def delete_issue(self, issue_id: int) -> bool:
        """
        Elimina un issue.

        DELETE /issues/{issue_id}.json

        Args:
            issue_id: ID del issue a eliminar.

        Returns:
            True si fue eliminado correctamente (HTTP 204).

        Raises:
            RedmineNotFoundError: Si el issue no existe (404).
            RedmineForbiddenError: Si no se tiene acceso (403).
            RedmineAPIError: Para cualquier otro error HTTP o de red.
        """
        return self._delete(f"/issues/{issue_id}.json")

    def add_watcher(self, issue_id: int, user_id: int) -> bool:
        """
        Añade un watcher a un issue.

        POST /issues/{issue_id}/watchers.json

        Args:
            issue_id: ID del issue.
            user_id: ID del usuario a añadir como watcher.

        Returns:
            True si el watcher fue añadido (HTTP 200).
        """
        self._post(
            f"/issues/{issue_id}/watchers.json",
            payload={"user_id": user_id},
        )
        return True

    def remove_watcher(self, issue_id: int, user_id: int) -> bool:
        """
        Elimina un watcher de un issue.

        DELETE /issues/{issue_id}/watchers/{user_id}.json

        Args:
            issue_id: ID del issue.
            user_id: ID del usuario a eliminar como watcher.

        Returns:
            True si el watcher fue eliminado (HTTP 204).
        """
        return self._delete(f"/issues/{issue_id}/watchers/{user_id}.json")

    # ------------------------------------------------------------------
    # Projects — https://www.redmine.org/projects/redmine/wiki/Rest_Projects
    # ------------------------------------------------------------------

    def list_projects(
        self,
        include: list[str] | None = None,
        limit: int = 25,
        offset: int = 0,
        fetch_all: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Lista todos los proyectos accesibles con la API key actual.

        GET /projects.json

        Args:
            include: Datos adicionales a incluir. Valores posibles:
                     "trackers", "issue_categories", "enabled_modules",
                     "time_entry_activities", "issue_custom_fields".
            limit: Proyectos por página (máx. 100).
            offset: Desplazamiento para paginación.
            fetch_all: Si es True, obtiene todos los proyectos iterando páginas.

        Returns:
            Lista de diccionarios con los proyectos.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if include:
            params["include"] = ",".join(include)

        if not fetch_all:
            data = self._get("/projects.json", params=params)
            return data.get("projects", [])

        all_projects: list[dict[str, Any]] = []
        current_offset = offset
        while True:
            params["offset"] = current_offset
            data = self._get("/projects.json", params=params)
            page = data.get("projects", [])
            all_projects.extend(page)
            total_count: int = data.get("total_count", 0)
            current_offset += len(page)
            if current_offset >= total_count or not page:
                break
        return all_projects

    def get_project(
        self,
        project_id: str | int,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Obtiene los detalles de un proyecto.

        GET /projects/{project_id}.json

        Args:
            project_id: ID numérico o identificador (string) del proyecto.
            include: Datos adicionales: "trackers", "issue_categories",
                     "enabled_modules", "time_entry_activities",
                     "issue_custom_fields".

        Returns:
            Diccionario con los datos del proyecto (campo ``project``).

        Raises:
            RedmineNotFoundError: Si el proyecto no existe (404).
            RedmineForbiddenError: Si no se tiene acceso (403).
        """
        params: dict[str, Any] = {}
        if include:
            params["include"] = ",".join(include)

        data = self._get(f"/projects/{project_id}.json", params=params)
        return data["project"]

    def create_project(
        self,
        name: str,
        identifier: str,
        description: str | None = None,
        homepage: str | None = None,
        is_public: bool | None = None,
        parent_id: int | None = None,
        inherit_members: bool | None = None,
        tracker_ids: list[int] | None = None,
        enabled_module_names: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Crea un nuevo proyecto.

        POST /projects.json

        Args:
            name: Nombre del proyecto (requerido).
            identifier: Identificador único del proyecto (requerido).
            description: Descripción del proyecto.
            homepage: URL de la página del proyecto.
            is_public: True para proyecto público, False para privado.
            parent_id: ID del proyecto padre.
            inherit_members: True para heredar miembros del padre.
            tracker_ids: Lista de IDs de trackers habilitados.
            enabled_module_names: Lista de módulos a activar
                (ej. ["issue_tracking", "time_tracking", "wiki"]).
            **kwargs: Campos adicionales.

        Returns:
            Diccionario con el proyecto creado (campo ``project``).

        Raises:
            RedmineValidationError: Si los datos son inválidos (422).
        """
        project_payload: dict[str, Any] = {
            "name": name,
            "identifier": identifier,
        }
        if description is not None:
            project_payload["description"] = description
        if homepage is not None:
            project_payload["homepage"] = homepage
        if is_public is not None:
            project_payload["is_public"] = is_public
        if parent_id is not None:
            project_payload["parent_id"] = parent_id
        if inherit_members is not None:
            project_payload["inherit_members"] = inherit_members
        if tracker_ids is not None:
            project_payload["tracker_ids"] = tracker_ids
        if enabled_module_names is not None:
            project_payload["enabled_module_names"] = enabled_module_names
        project_payload.update(kwargs)

        data = self._post("/projects.json", payload={"project": project_payload})
        return data["project"]

    def update_project(
        self,
        project_id: str | int,
        **kwargs: Any,
    ) -> bool:
        """
        Actualiza un proyecto existente.

        PUT /projects/{project_id}.json

        Args:
            project_id: ID numérico o identificador del proyecto.
            **kwargs: Campos a actualizar (name, description, is_public, etc.).

        Returns:
            True si la actualización fue exitosa (HTTP 204).
        """
        return self._put(
            f"/projects/{project_id}.json",
            payload={"project": kwargs},
        )

    def delete_project(self, project_id: str | int) -> bool:
        """
        Elimina un proyecto.

        DELETE /projects/{project_id}.json

        Args:
            project_id: ID numérico o identificador del proyecto.

        Returns:
            True si fue eliminado correctamente (HTTP 204).
        """
        return self._delete(f"/projects/{project_id}.json")

    # ------------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------------

    def _handle_response_error(self, response: httpx.Response) -> None:
        """Convierte errores HTTP en excepciones tipadas."""
        if response.status_code == 404:
            raise RedmineNotFoundError(f"Recurso no encontrado: {response.request.url}")
        if response.status_code == 403:
            raise RedmineForbiddenError(f"Acceso denegado: {response.request.url}")
        if response.status_code == 422:
            try:
                errors: list[str] = response.json().get("errors", [])
            except Exception:
                errors = [response.text]
            raise RedmineValidationError(errors)
        response.raise_for_status()

    def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Petición GET genérica con manejo de errores.

        Args:
            endpoint: Path relativo (ej. "/issues.json").
            params: Query parameters opcionales.

        Returns:
            Respuesta JSON deserializada como dict.
        """
        # Eliminar params con valor None para no enviarlos en la query string
        clean_params = {k: v for k, v in (params or {}).items() if v is not None}
        try:
            response = self._client.get(endpoint, params=clean_params)
            self._handle_response_error(response)
            return response.json()
        except (
            RedmineNotFoundError,
            RedmineForbiddenError,
            RedmineValidationError,
            RedmineAPIError,
        ):
            raise
        except httpx.HTTPStatusError as e:
            raise RedmineAPIError(f"HTTP {e.response.status_code}: {e.response.text}") from e
        except httpx.RequestError as e:
            raise RedmineAPIError(f"Error de conexión con Redmine: {e}") from e

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Petición POST genérica.

        Args:
            endpoint: Path relativo (ej. "/issues.json").
            payload: Cuerpo de la petición como dict (será serializado a JSON).

        Returns:
            Respuesta JSON deserializada como dict (puede ser {} si no hay body).
        """
        try:
            response = self._client.post(endpoint, json=payload)
            self._handle_response_error(response)
            # 201 Created puede incluir el recurso creado
            if response.content:
                return response.json()
            return {}
        except (
            RedmineNotFoundError,
            RedmineForbiddenError,
            RedmineValidationError,
            RedmineAPIError,
        ):
            raise
        except httpx.HTTPStatusError as e:
            raise RedmineAPIError(f"HTTP {e.response.status_code}: {e.response.text}") from e
        except httpx.RequestError as e:
            raise RedmineAPIError(f"Error de conexión con Redmine: {e}") from e

    def _put(self, endpoint: str, payload: dict[str, Any]) -> bool:
        """
        Petición PUT genérica. Redmine devuelve 204 No Content en éxito.

        Args:
            endpoint: Path relativo (ej. "/issues/42.json").
            payload: Cuerpo de la petición como dict.

        Returns:
            True si la operación fue exitosa (HTTP 204).
        """
        try:
            response = self._client.put(endpoint, json=payload)
            self._handle_response_error(response)
            return response.status_code == 204
        except (
            RedmineNotFoundError,
            RedmineForbiddenError,
            RedmineValidationError,
            RedmineAPIError,
        ):
            raise
        except httpx.HTTPStatusError as e:
            raise RedmineAPIError(f"HTTP {e.response.status_code}: {e.response.text}") from e
        except httpx.RequestError as e:
            raise RedmineAPIError(f"Error de conexión con Redmine: {e}") from e

    def _delete(self, endpoint: str) -> bool:
        """
        Petición DELETE genérica. Redmine devuelve 204 No Content en éxito.

        Args:
            endpoint: Path relativo (ej. "/issues/42.json").

        Returns:
            True si el recurso fue eliminado (HTTP 204).
        """
        try:
            response = self._client.delete(endpoint)
            self._handle_response_error(response)
            return response.status_code == 204
        except (
            RedmineNotFoundError,
            RedmineForbiddenError,
            RedmineValidationError,
            RedmineAPIError,
        ):
            raise
        except httpx.HTTPStatusError as e:
            raise RedmineAPIError(f"HTTP {e.response.status_code}: {e.response.text}") from e
        except httpx.RequestError as e:
            raise RedmineAPIError(f"Error de conexión con Redmine: {e}") from e
