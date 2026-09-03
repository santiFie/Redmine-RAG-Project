from typing import Annotated, Any, Literal, NotRequired
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class BugReportExtraction(BaseModel):
    is_sufficient: bool = Field(
        description="True si el usuario proporcionó suficiente contexto del error para reproducirlo o categorizarlo."
    )
    missing_fields: list[str] = Field(
        default_factory=list,
        description="Lista de datos ausentes necesarios (ej. pasos para reproducir, entorno, versión, mensaje de error exacto).",
    )
    title_summary: str = Field(
        description="Título conciso y profesional para el posible ticket de Redmine."
    )
    reproduction_steps: str = Field(
        default="", description="Pasos para reproducir reconstruidos a partir de la conversación."
    )
    environment_info: str = Field(
        default="", description="Detalles del entorno (navegador, SO, módulo del sistema)."
    )
    suggested_tracker: Literal["Bug", "Feature", "Soporte"] = Field(default="Bug")
    suggested_priority: Literal["Baja", "Normal", "Alta", "Urgente"] = Field(default="Normal")


class State(TypedDict):
    # Historial de mensajes y entradas
    messages: Annotated[list[Any], add_messages]
    user_input: str
    is_safe_query: bool
    intent: Literal["knowledge_query", "incident_report", "general"]

    # Rama RAG Convencional
    rag_context: NotRequired[str]
    retrieved_issue_ids: NotRequired[list[int]]

    # Rama QA & Triaje
    bug_analysis: NotRequired[BugReportExtraction | None]
    is_duplicate: NotRequired[bool]
    duplicate_issue_id: NotRequired[int | None]
    suggested_project_id: NotRequired[str | None]
    target_project_id: NotRequired[str | None]
    available_projects: NotRequired[list[dict[str, Any]]]
    created_issue_id: NotRequired[int | None]
    created_issue_url: NotRequired[str | None]

    # Salida Final
    final_answer: str
