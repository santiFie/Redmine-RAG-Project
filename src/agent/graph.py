"""
src/agent/graph.py
==================
Grafo principal de LangGraph para el agente Redmine + RAG.

Flujo:
  START → analyze_intent → [route]
                         ├─ redmine_agent  (tools del MCP de Redmine)
                         └─ rag_query      (retrieval LlamaIndex)
                                 └─ respond (LLM unificado) → END

Decisión de diseño — nodo `respond` (Opción B):
  LlamaIndex SOLO hace el retrieval y deposita el contexto en
  state["rag_context"]. El nodo `respond` es quien construye el
  prompt final y llama al LLM, lo que permite:
    • Gestión consistente del historial de mensajes (add_messages).
    • Posibilidad de combinar contexto RAG + resultado de Redmine
      en una respuesta cohesiva.
    • Mayor control sobre el tono y formato de la respuesta.
"""

from __future__ import annotations
from src.agent.state import BugReportExtraction, State
from pathlib import Path
from typing import Annotated, Any, Literal, NotRequired

from dotenv import load_dotenv
from guardrails import Guard
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

import asyncio
import logging
import os
from src.prompts import aget_prompt
from src.rag.engine import RAGEngine
from src.redmine.client import RedmineClient, RedmineAPIError
from src.utils.get_llm import get_llm

logger = logging.getLogger(__name__)

load_dotenv()


# ==============================================================================
# Clasificación de seguridad (safe-query)
# ==============================================================================


class SafeQueryClassification(BaseModel):
    reasoning: str = Field(
        description="Breve razonamiento paso a paso de por qué elegiste esa intención"
    )
    is_safe_query: bool


async def analyze_safe_query_node(state: State) -> dict[str, Any]:
    """
    Clasifica si la consulta es segura para enviar al RAG o Redmine.
    """
    messages = state.get("messages", [])
    last_msg = messages[-1] if messages else None

    user_input = state.get("user_input")
    if not user_input and last_msg:
        # user_input is in messages
        user_input = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    llm = get_llm("groq", "openai/gpt-oss-20b", 0.0)
    structured_llm = llm.with_structured_output(SafeQueryClassification)

    prompt = await aget_prompt("analyze-safe-query")
    messages = prompt.format_messages(user_input=user_input)

    response = await structured_llm.ainvoke(messages)

    output_state = {
        "user_input": user_input,
        "is_safe_query": response.is_safe_query,
    }

    if not messages and user_input:
        # add user input to messages (ground truth)
        output_state["messages"] = [HumanMessage(content=user_input)]

    return output_state


# ==============================================================================
# Clasificación de intención
# ==============================================================================


class IntentClassification(BaseModel):
    reasoning: str = Field(description="Explicación breve de por qué se eligió la intención.")
    intent: Literal["knowledge_query", "incident_report", "general"]


def route_after_analyze(state: dict[str, Any]) -> str:
    """Función de enrutamiento mantenida para compatibilidad con tests unitarios."""
    intent = state.get("intent", "general")
    if intent in ("rag_query", "knowledge_query"):
        return "rag_query"
    elif intent in ("redmine_mcp", "incident_report"):
        return "redmine_agent"
    return "respond_general"


async def analyze_intent_node(state: State) -> State:
    """
    Clasifica la intención del mensaje del usuario.

    Salida al estado: state["intent"], state["user_input"]
    """
    user_input: str = state["user_input"]
    llm = get_llm("groq", "openai/gpt-oss-20b", 0.1)
    structured_llm = llm.with_structured_output(IntentClassification)

    prompt = await aget_prompt("analyze-intent")
    messages = prompt.format_messages(user_input=user_input)

    response = await structured_llm.ainvoke(messages)

    return {
        "intent": response.intent,
    }


# ==============================================================================
# Rama 1: Consulta a la documentación (Knowledge Agent)
# ==============================================================================

# Rutas resueltas una sola vez al importar el módulo.
# Path(__file__) apunta a src/agent/graph.py → .parent.parent.parent = raíz del proyecto.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_MCP_SERVER = _PROJECT_ROOT / "mcp" / "redmine" / "server.py"
_PYTHON_EXE = _PROJECT_ROOT / ".venv" / "bin" / "python"

_MCP_CONFIG = {
    "redmine": {
        "command": str(_PYTHON_EXE),
        "args": [str(_MCP_SERVER)],
        "transport": "stdio",
        "env": {
            "REDMINE_URL": os.getenv("REDMINE_URL", ""),
            "REDMINE_API_KEY": os.getenv("REDMINE_API_KEY", ""),
            "PYTHONPATH": str(_PROJECT_ROOT / "mcp" / "redmine"),
        },
    }
}


async def get_redmine_tools(allowed_names: set[str] | None = None) -> list[BaseTool]:
    """Carga herramientas del servidor MCP de Redmine, filtrando opcionalmente por nombre."""
    client = MultiServerMCPClient(_MCP_CONFIG)
    tools = await client.get_tools(server_name="redmine")
    if allowed_names is not None:
        tools = [t for t in tools if t.name in allowed_names]
    return tools


def make_redmine_agent_node():
    """
    Construye el nodo ejecutable del redmine_agent utilizando un sub-grafo
    con StateGraph, ToolNode y tools_condition.
    Esto permite un control estricto de ejecuciones de herramientas MCP
    y evita el consumo desmedido de tokens en el proveedor de LLM.
    """

    llm = get_llm("gemini", "gemini-2.5-flash", 0.0)

    async def redmine_agent(state: State) -> State:
        """
        Nodo async que crea el cliente MCP, carga las herramientas y ejecuta
        un sub-grafo con ToolNode y tools_condition para interactuar con Redmine.
        """
        client = MultiServerMCPClient(_MCP_CONFIG)

        async with client.session("redmine") as session:
            tools = await load_mcp_tools(session)
            llm_with_tools = llm.bind_tools(tools)

            prompt = await aget_prompt("redmine-agent")
            # La template de redmine-agent solo tiene un SystemMessage, lo extraemos:
            sys_msg = prompt.format_messages()[0]

            # Sub-grafo con StateGraph, ToolNode y tools_condition
            sub_builder = StateGraph(State)

            async def call_model(sub_state: State) -> dict[str, Any]:
                response = await llm_with_tools.ainvoke(sub_state["messages"])
                return {"messages": [response]}

            tool_node = ToolNode(tools)

            sub_builder.add_node("agent", call_model)
            sub_builder.add_node("tools", tool_node)

            sub_builder.add_edge(START, "agent")
            sub_builder.add_conditional_edges(
                "agent",
                tools_condition,
                {
                    "tools": "tools",
                    END: END,
                },
            )
            sub_builder.add_edge("tools", "agent")

            sub_graph = sub_builder.compile()

            # Invocación del sub-grafo dentro del contexto de sesión del MCP
            input_messages = [sys_msg] + list(state["messages"])
            sub_result = await sub_graph.ainvoke({"messages": input_messages})

        last_ai_msg = next(
            (m for m in reversed(sub_result["messages"]) if isinstance(m, AIMessage) and m.content),
            None,
        )
        agent_response = (
            last_ai_msg.content if last_ai_msg else "Sin respuesta del agente de Redmine."
        )

        return {
            "redmine_result": agent_response,
            "final_answer": agent_response,
            "messages": [AIMessage(content=agent_response)],
        }

    return redmine_agent


# ==============================================================================
# Nodo 3: RAG Query (LlamaIndex retrieval — solo recupera contexto)
# ==============================================================================


# RAGEngine se inicializa en tiempo de importación del módulo (startup del servidor),
# antes de que el event loop de LangGraph esté activo.
# Esto evita que el I/O bloqueante de QdrantVectorStore.__init__ ocurra
# durante la ejecución de un nodo async.
_rag_engine: RAGEngine = RAGEngine()


def get_rag_engine() -> RAGEngine:
    return _rag_engine


def _rag_query_sync(question: str) -> dict[str, Any]:
    """Mantenido para compatibilidad con tests. No se usa en el grafo."""
    return get_rag_engine().query(question)


async def rag_query_node(state: State) -> dict[str, Any]:
    user_question = state["user_input"]

    # Búsqueda vectorial async nativa — no bloquea el event loop de LangGraph
    result = await get_rag_engine().aquery(user_question)

    return {"rag_context": result["context"], "retrieved_issue_ids": result["issue_ids"]}


# ==============================================================================
# Nodo 4: Respond — LLM unificado con contexto RAG
# ==============================================================================


async def respond_knowledge_node(state: State) -> State:
    """
    Genera la respuesta final al usuario combinando:
      - state["rag_context"]: contexto recuperado por LlamaIndex
      - state["messages"]:    historial de conversación

    Se usa para la rama de búsqueda de documentos (search_docs).
    Para intenciones de Redmine, el redmine_agent ya escribe en messages.

    Input del estado : state["rag_context"], state["messages"]
    Output al estado : state["messages"] (respuesta añadida), state["final_answer"]
    """
    llm = get_llm("groq", "openai/gpt-oss-120b", 0.3)

    # Extraer la pregunta original del usuario
    user_question = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        state.get("user_input", ""),
    )

    rag_context = state.get("rag_context", "")

    if not rag_context:
        raise ValueError(
            "No se recuperó contexto del RAG. Revise si existe contexto indexado o si la pregunta es válida."
        )

    prompt = await aget_prompt("respond-rag")
    messages = prompt.format_messages(rag_context=rag_context, user_input=user_question)

    issue_ids = state.get("retrieved_issue_ids", [])
    if issue_ids:
        base_url = os.getenv("REDMINE_HOST_URL", "").rstrip("/")
        links_instruction = "IMPORTANTE: Al final de tu respuesta, debes incluir EXACTAMENTE esta sección de referencias (sin modificar los links):\n\n### 🔗 Tickets de Referencia\n"
        for iid in issue_ids:
            links_instruction += f"- [Ticket #{iid}]({base_url}/issues/{iid})\n"

        # Agregamos la instrucción antes de llamar al LLM
        messages.append(SystemMessage(content=links_instruction))

    response = await llm.ainvoke(messages)
    answer = response.content

    return {
        "messages": [AIMessage(content=answer)],
        "final_answer": answer,
    }


# ==============================================================================
# Nodo 5: Respond general (sin RAG — para intenciones "general")
# ==============================================================================


async def respond_general_node(state: State) -> State:
    """
    Responde a intenciones generales (saludos, preguntas simples)
    sin necesidad de RAG ni Redmine.
    """
    llm = get_llm("nvidia", "deepseek-ai/deepseek-v4-flash-0731", 0.5)

    prompt = await aget_prompt("respond-general")
    sys_msg = prompt.format_messages()[0]

    response = await llm.ainvoke([sys_msg] + list(state["messages"]))
    answer = response.content

    return {
        "messages": [AIMessage(content=answer)],
        "final_answer": answer,
    }


# ==============================================================================
# Nodo 6: Guardrail de Salida (Guardrails AI)
# ==============================================================================


async def output_guardrail_node(state: State) -> State:
    """
    Nodo de guardrail de salida usando Guardrails AI.
    Valida la respuesta generada por el LLM/agente antes de entregarla al usuario.
    """
    content = state.get("final_answer", "")

    if not content:
        return state

    # Parsear y validar con Guardrails AI
    guard = Guard()
    try:
        outcome = guard.parse(llm_output=content)
        if outcome.validation_passed:
            validated_text = outcome.validated_output or content
        else:
            validated_text = (
                "Lo siento, la respuesta generada no superó los controles de seguridad y calidad."
            )
    except Exception:
        validated_text = content

    return {
        "final_answer": validated_text,
    }


# ==============================================================================
# QA Evalator Node
# ==============================================================================


async def qa_evaluator_node(state: State) -> State:
    user_input = state["user_input"]

    llm = get_llm("nvidia", "deepseek-ai/deepseek-v4-flash-0731", 0.1)
    structured_llm = llm.with_structured_output(BugReportExtraction)

    prompt = await aget_prompt("qa-evaluator")
    messages = prompt.format_messages(question=user_input)

    response = await structured_llm.ainvoke(messages)

    return {
        "bug_analysis": response,
    }


# ==============================================================================
# Ask Clarification Node
# ==============================================================================


async def ask_clarification_node(state: State) -> dict[str, Any]:
    bug_analysis = state.get("bug_analysis")
    missing_fields = bug_analysis.missing_fields if bug_analysis else []
    clarification_questions = bug_analysis.clarification_questions if bug_analysis else []
    user_input = state.get("user_input", "")

    # Construir bloque contextualizado para el prompt
    if clarification_questions:
        formatted_items = "\n".join(f"- {q}" for q in clarification_questions)
    elif missing_fields:
        formatted_items = "\n".join(f"- {f}" for f in missing_fields)
    else:
        formatted_items = "- Contexto adicional o detalles sobre el comportamiento observado."

    contextual_missing = (
        f'Mensaje del usuario:\n"""{user_input}"""\n\n'
        f"Puntos o preguntas a clarificar:\n{formatted_items}"
    )

    llm = get_llm("nvidia", "deepseek-ai/deepseek-v4-flash-0731", 0.1)

    prompt = await aget_prompt("ask-clarification")
    messages = prompt.format_messages(missing_fields=contextual_missing)

    response = await llm.ainvoke(messages)
    question_text = response.content

    user_response = interrupt(
        {
            "action": "provide_clarification",
            "question": question_text,
            "missing_fields": missing_fields,
            "clarification_questions": clarification_questions,
        }
    )

    current_turns = state.get("clarification_turns", 0) + 1
    updated_user_input = (
        f"{user_input}\n\n[Información adicional provista por el usuario]:\n{user_response}"
    )

    return {
        "messages": [
            AIMessage(content=question_text),
            HumanMessage(content=str(user_response)),
        ],
        "user_input": updated_user_input,
        "clarification_turns": current_turns,
    }


class DuplicateCheckResult(BaseModel):
    is_duplicate: bool = Field(
        description="True si el problema es un duplicado abierto o tiene una solución histórica idéntica."
    )
    duplicate_issue_id: int | None = Field(
        default=None, description="ID del ticket duplicado o con la solución, si aplica."
    )
    reasoning: str = Field(
        description="Justificación de la decisión basándose en el contexto recuperado o proyectos analizados."
    )
    suggested_project_id: str | None = Field(
        default=None,
        description="Identificador (identifier o id) del proyecto de Redmine más adecuado para crear el issue, elegido de los proyectos disponibles.",
    )


async def duplicate_and_rag_check_node(state: State) -> dict[str, Any]:
    bug_analysis = state.get("bug_analysis")
    if not bug_analysis:
        return {}

    # Construir Query Enriquecida
    query_parts = [f"Problema: {bug_analysis.title_summary}"]
    if getattr(bug_analysis, "incident_type", None):
        query_parts.append(f"Tipo: {bug_analysis.incident_type}")
    if getattr(bug_analysis, "technical_details", ""):
        query_parts.append(f"Detalles: {bug_analysis.technical_details}")
    if bug_analysis.environment_info:
        query_parts.append(f"Entorno: {bug_analysis.environment_info}")
    if bug_analysis.reproduction_steps:
        query_parts.append(f"Pasos: {bug_analysis.reproduction_steps}")

    enriched_query = "\n".join(query_parts)

    # 1. Recuperar contexto vía RAG
    rag_context = ""
    retrieved_ids = []
    try:
        result = await get_rag_engine().aquery(enriched_query)
        rag_context = result.get("context", "")
        retrieved_ids = result.get("issue_ids", [])
    except Exception as e:
        logger.warning(f"Error al consultar el motor RAG en duplicate check: {e}")

    # 2. Obtener catálogo de proyectos de Redmine de forma no bloqueante
    def _fetch_projects() -> list[dict[str, Any]]:
        try:
            client = RedmineClient()
            return client.list_projects()
        except Exception as e:
            logger.warning(f"No se pudieron recuperar proyectos de Redmine: {e}")
            return []

    available_projects = await asyncio.to_thread(_fetch_projects)

    if available_projects:
        projects_text = "\n".join(
            [
                f"- ID/Identifier: '{p.get('identifier') or p.get('id')}' | Nombre: '{p.get('name')}' | Descripción: '{p.get('description', '')[:100]}'"
                for p in available_projects
            ]
        )
    else:
        default_proj = os.getenv("REDMINE_DEFAULT_PROJECT", "test-project")
        projects_text = f"- ID/Identifier: '{default_proj}' | Nombre: 'Proyecto por defecto'"

    # 3. Evaluar duplicados y deducir proyecto con LLM
    llm = get_llm("nvidia", "deepseek-ai/deepseek-v4-flash-0731", 0.1)
    structured_llm = llm.with_structured_output(DuplicateCheckResult)

    prompt_template = await aget_prompt("triage-duplicate-check")
    formatted_rag_context = (
        rag_context
        if rag_context.strip()
        else "No se encontraron tickets similares en la base de conocimiento (similitud baja o nula)."
    )
    messages = prompt_template.format_messages(
        enriched_query=enriched_query,
        rag_context=formatted_rag_context,
        projects_text=projects_text,
    )

    eval_result = await structured_llm.ainvoke(messages)

    suggested_proj = eval_result.suggested_project_id
    if not suggested_proj:
        if available_projects:
            suggested_proj = str(
                available_projects[0].get("identifier") or available_projects[0].get("id")
            )
        else:
            suggested_proj = os.getenv("REDMINE_DEFAULT_PROJECT", "test-project")

    return {
        "is_duplicate": eval_result.is_duplicate,
        "duplicate_issue_id": eval_result.duplicate_issue_id,
        "suggested_project_id": suggested_proj,
        "available_projects": available_projects,
        "rag_context": rag_context,
        "retrieved_issue_ids": retrieved_ids,
    }


async def confirm_issue_creation_node(state: State) -> dict[str, Any]:
    """
    Pausa el flujo (HITL) para que el usuario confirme o seleccione el proyecto
    de Redmine donde se creará el ticket antes de proceder a la creación.
    """
    bug_analysis = state.get("bug_analysis")
    suggested_project = state.get("suggested_project_id") or os.getenv(
        "REDMINE_DEFAULT_PROJECT", "test-project"
    )
    available_projects = state.get("available_projects", [])

    project_options = [
        {"id": p.get("id"), "identifier": p.get("identifier"), "name": p.get("name")}
        for p in available_projects
    ]

    question_text = (
        f"Se creará un nuevo ticket para el incidente '{bug_analysis.title_summary if bug_analysis else 'Incidente'}' "
        f"en el proyecto '{suggested_project}'. ¿Deseas confirmar este proyecto o seleccionar otro?"
    )

    user_response = interrupt(
        {
            "action": "confirm_issue_creation",
            "question": question_text,
            "suggested_project": suggested_project,
            "available_projects": project_options,
        }
    )

    selected_project = suggested_project
    if isinstance(user_response, dict) and user_response.get("project_id"):
        selected_project = str(user_response["project_id"])
    elif isinstance(user_response, str) and user_response.strip():
        resp_clean = user_response.strip()
        if resp_clean.lower() not in {"si", "sí", "yes", "confirmar", "ok", "confirmo"}:
            selected_project = resp_clean

    return {
        "target_project_id": selected_project,
        "messages": [
            AIMessage(content=question_text),
            HumanMessage(content=str(user_response)),
        ],
    }


async def redmine_issue_creator_node(state: State) -> dict[str, Any]:
    """
    Crea el issue directamente mediante RedmineClient (API REST) usando el proyecto confirmado.
    """
    bug_analysis = state.get("bug_analysis")
    if not bug_analysis:
        return {}

    target_project = (
        state.get("target_project_id")
        or state.get("suggested_project_id")
        or os.getenv("REDMINE_DEFAULT_PROJECT", "test-project")
    )

    if getattr(bug_analysis, "description_markdown", "").strip():
        description = bug_analysis.description_markdown
    else:
        description = (
            f"h3. Pasos para Reproducir\n{bug_analysis.reproduction_steps}\n\n"
            f"h3. Entorno\n{bug_analysis.environment_info}"
        )

    def _create() -> dict[str, Any]:
        client = RedmineClient()
        return client.create_issue(
            project_id=target_project,
            subject=bug_analysis.title_summary,
            description=description,
        )

    issue_data = await asyncio.to_thread(_create)
    issue_id = issue_data["id"]

    base_url = (os.getenv("REDMINE_URL")).rstrip("/")
    issue_url = f"{base_url}/issues/{issue_id}"

    return {
        "created_issue_id": issue_id,
        "created_issue_url": issue_url,
    }


async def respond_existing_solution_node(state: State) -> State:
    dup_id = state.get("duplicate_issue_id")
    msg = f"He encontrado que este problema ya está documentado o reportado en el ticket #{dup_id}."
    return {"final_answer": msg, "messages": [AIMessage(content=msg)]}


async def respond_creation_summary_node(state: State) -> State:
    url = state.get("created_issue_url", "#")
    msg = f"He creado un nuevo ticket para este incidente: {url}"
    return {"final_answer": msg, "messages": [AIMessage(content=msg)]}


def route_qa(state: State) -> Literal["ask_clarification", "duplicate_and_rag_check"]:
    """Enruta el flujo QA a verificación de duplicados o clarificación, evitando bucles."""
    analysis = state.get("bug_analysis")
    # Si ya se realizó al menos 1 turno de clarificación, no ciclar indefinidamente
    if state.get("clarification_turns", 0) >= 1:
        return "duplicate_and_rag_check"
    if analysis and analysis.is_sufficient:
        return "duplicate_and_rag_check"
    return "ask_clarification"


# ==============================================================================
# Construcción del Grafo
# ==============================================================================


def build_graph() -> StateGraph:
    workflow = StateGraph(State)

    # ── 1. Registro de Nodos ──────────────────────────────────────────────────
    workflow.add_node("analyze_safe_query", analyze_safe_query_node)
    workflow.add_node("analyze_intent", analyze_intent_node)

    # Rama RAG
    workflow.add_node("rag_knowledge_query", rag_query_node)
    workflow.add_node("respond_knowledge", respond_knowledge_node)

    # Rama QA & Triaje
    workflow.add_node("qa_evaluator", qa_evaluator_node)
    workflow.add_node("ask_clarification", ask_clarification_node)
    workflow.add_node("duplicate_and_rag_check", duplicate_and_rag_check_node)
    workflow.add_node("confirm_creation", confirm_issue_creation_node)
    workflow.add_node("redmine_creator", redmine_issue_creator_node)
    workflow.add_node("respond_existing", respond_existing_solution_node)
    workflow.add_node("respond_creation", respond_creation_summary_node)

    # Rama General & Guardrails
    workflow.add_node("respond_general", respond_general_node)
    workflow.add_node("output_guardrail", output_guardrail_node)

    # ── 2. Edges y Enrutamiento ───────────────────────────────────────────────
    workflow.add_edge(START, "analyze_safe_query")

    def route_safe_query(state: State) -> Literal["analyze_intent", "__end__"]:
        return "analyze_intent" if state.get("is_safe_query", True) else "__end__"

    workflow.add_conditional_edges("analyze_safe_query", route_safe_query)

    def route_intent(
        state: State,
    ) -> Literal["rag_knowledge_query", "qa_evaluator", "respond_general"]:
        intent = state.get("intent", "general")
        if intent == "knowledge_query":
            return "rag_knowledge_query"
        elif intent == "incident_report":
            return "qa_evaluator"
        return "respond_general"

    workflow.add_conditional_edges("analyze_intent", route_intent)

    # Conexiones Rama RAG
    workflow.add_edge("rag_knowledge_query", "respond_knowledge")
    workflow.add_edge("respond_knowledge", "output_guardrail")

    # Conexiones Rama QA
    workflow.add_conditional_edges("qa_evaluator", route_qa)
    workflow.add_edge("ask_clarification", "qa_evaluator")

    def route_duplicates(state: State) -> Literal["respond_existing", "confirm_creation"]:
        if state.get("is_duplicate", False):
            return "respond_existing"
        return "confirm_creation"

    workflow.add_conditional_edges("duplicate_and_rag_check", route_duplicates)
    workflow.add_edge("confirm_creation", "redmine_creator")
    workflow.add_edge("redmine_creator", "respond_creation")
    workflow.add_edge("respond_existing", "output_guardrail")
    workflow.add_edge("respond_creation", "output_guardrail")

    # Conexiones Rama General y Salida
    workflow.add_edge("respond_general", "output_guardrail")
    workflow.add_edge("output_guardrail", END)

    return workflow


workflow = build_graph()
graph = workflow.compile()
