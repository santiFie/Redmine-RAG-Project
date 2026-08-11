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

import asyncio
import os
from typing import Annotated, Any, Literal

from dotenv import load_dotenv
from guardrails import Guard
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent  # type: ignore[reportDeprecated]
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from src.rag.engine import RAGEngine
from src.utils.get_llm import get_llm

load_dotenv()

# ==============================================================================
# Estado del Agente
# ==============================================================================

class AgentState(TypedDict):
    messages: Annotated[list[Any], add_messages]
    user_input: str | None
    is_safe_query: bool
    intent: Literal["query_issue", "create_issue", "update_issue", "list_issues", "search_docs", "general"]
    redmine_result: str          # resultado serializado del redmine_agent
    rag_context: str             # contexto recuperado por LlamaIndex
    final_answer: str
    error: str | None


# ==============================================================================
# Nodo 0: Clasificación de intención
# ==============================================================================

class SafeQueryClassification(BaseModel):
    reasoning: str = Field(description="Breve razonamiento paso a paso de por qué elegiste esa intención")
    is_safe_query: bool


async def analyze_safe_query(state: AgentState) -> AgentState:
    """
    Clasifica si la consulta es segura para enviar al RAG o Redmine.
    """
    user_input = state["user_input"]
    llm = get_llm("huggingface", "meta-llama/Llama-Guard-3-8B:featherless-ai", 0.0)
    structured_llm = llm.with_structured_output(SafeQueryClassification)

    sys_msg = SystemMessage(
        content=(
            "Eres un clasificador de consultas seguras. "
            "Tu objetivo es determinar si la consulta del usuario es segura para enviar al RAG o Redmine.\n\n"
            "Consultas seguras incluyen: búsqueda de documentos, consulta de issues, etc.\n"
            "Consultas inseguras incluyen: información personal, financiera, etc.\n\n"
            "Si la consulta es insegura, responde con false.\n"
            "Si la consulta es segura, responde con true.\n"
        )
    )

    response = await structured_llm.ainvoke([sys_msg, user_input])

    return {
        "user_input": user_input,
        "is_safe_query": response.is_safe_query,
    }

# ==============================================================================
# Nodo 1: Clasificación de intención
# ==============================================================================

class IntentClassification(BaseModel):
    reasoning: str = Field(description="Breve razonamiento paso a paso de por qué elegiste esa intención")
    intent: Literal["query_issue", "create_issue", "update_issue", "list_issues", "search_docs", "general"]


async def analyze_intent(state: AgentState) -> AgentState:
    """
    Clasifica la intención del último mensaje del usuario.

    Salida al estado: state["intent"], state["user_input"]
    """
    user_input = state["user_input"]
    llm = get_llm("groq", "llama-3.3-70b-versatile", 0.1)
    structured_llm = llm.with_structured_output(IntentClassification)

    sys_msg = SystemMessage(
        content=(
            "Eres el enrutador de un orquestador de servicios. "
            "Tu objetivo es clasificar la intención del usuario a partir de su input.\n\n"
            "Intenciones disponibles:\n"
            "- query_issue: búsqueda de un issue específico (ej. estado, detalles).\n"
            "- create_issue: creación de un nuevo issue.\n"
            "- update_issue: actualización de un issue existente.\n"
            "- list_issues: listado general o búsqueda filtrada de issues.\n"
            "- search_docs: búsqueda de información técnica o manuales en RAG.\n"
            "- general: saludos o conversación general que no requiere herramientas.\n"
        )
    )

    response = await structured_llm.ainvoke([sys_msg, user_input])

    return {
        "user_input": user_input,
        "intent": response.intent,
    }


# ==============================================================================
# Nodo 2: Redmine Agent (via MCP)
# ==============================================================================

def make_redmine_agent_node():
    """
    Construye el nodo del redmine_agent que usa las herramientas del MCP server
    de Redmine a través de langchain-mcp-adapters (stdio transport).

    El server MCP se lanza como subproceso usando el Dockerfile de mcp/redmine.
    El agente es un ReAct loop que puede usar múltiples herramientas en secuencia.
    """
    mcp_server_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "mcp", "redmine", "server.py"
    )
    python_exe = os.path.join(
        os.path.dirname(__file__), "..", "..", ".venv", "bin", "python"
    )

    # Configuración del cliente MCP (stdio — lanza el server como subproceso)
    mcp_config = {
        "redmine": {
            "command": os.path.abspath(python_exe),
            "args": [os.path.abspath(mcp_server_path)],
            "transport": "stdio",
            "env": {
                "REDMINE_URL": os.getenv("REDMINE_URL", ""),
                "REDMINE_API_KEY": os.getenv("REDMINE_API_KEY", ""),
                "PYTHONPATH": os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..", "..", "mcp", "redmine")
                ),
            },
        }
    }

    llm = get_llm("groq", "llama-3.3-70b-versatile", 0.0)

    async def redmine_agent(state: AgentState) -> AgentState:
        """
        Nodo async que crea el cliente MCP, obtiene las tools y ejecuta
        un ReAct agent para resolver la solicitud del usuario sobre Redmine.

        Input del estado : state["messages"], state["user_input"]
        Output al estado : state["redmine_result"], state["messages"]
        """
        async with MultiServerMCPClient(mcp_config) as client:
            tools = client.get_tools()

            agent = create_react_agent(llm, tools)

            sys_msg = SystemMessage(
                content=(
                    "Eres un asistente de gestión de proyectos con acceso a Redmine. "
                    "Usa las herramientas disponibles para responder la solicitud del usuario. "
                    "Si necesitás más de una herramienta, usálas en secuencia. "
                    "Respondé siempre en el idioma del usuario."
                )
            )

            # Invocar el ReAct agent con el historial completo
            result = await agent.ainvoke({
                "messages": [sys_msg] + list(state["messages"])
            })

            # Extraer la última respuesta del agente
            last_ai_msg = next(
                (m for m in reversed(result["messages"]) if isinstance(m, AIMessage)),
                None,
            )
            agent_response = last_ai_msg.content if last_ai_msg else "Sin respuesta del agente."

            return {
                "redmine_result": agent_response,
                "messages": [AIMessage(content=agent_response)],
            }

    return redmine_agent


# ==============================================================================
# Nodo 3: RAG Query (LlamaIndex retrieval — solo recupera contexto)
# ==============================================================================

_rag_engine: RAGEngine | None = None


def get_rag_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine


def _rag_query_sync(question: str) -> str:
    return get_rag_engine().query(question)


async def rag_query(state: AgentState):
    user_question = state["user_input"]

    # Construcción + consulta síncrona de LlamaIndex aisladas en un Worker Thread
    context = await asyncio.to_thread(_rag_query_sync, user_question)

    return {"rag_context": context}

# ==============================================================================
# Nodo 4: Respond — LLM unificado con contexto RAG
# ==============================================================================

async def respond(state: AgentState) -> AgentState:
    """
    Genera la respuesta final al usuario combinando:
      - state["rag_context"]: contexto recuperado por LlamaIndex
      - state["messages"]:    historial de conversación

    Se usa para la rama de búsqueda de documentos (search_docs).
    Para intenciones de Redmine, el redmine_agent ya escribe en messages.

    Input del estado : state["rag_context"], state["messages"]
    Output al estado : state["messages"] (respuesta añadida), state["final_answer"]
    """
    llm = get_llm("groq", "llama-3.3-70b-versatile", 0.3)

    # Extraer la pregunta original del usuario
    user_question = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        state.get("user_input", ""),
    )

    rag_context = state.get("rag_context", "")

    if not rag_context:
        raise ValueError("No se recuperó contexto del RAG. Revise si existe contexto indexado o si la pregunta es válida.")

    sys_msg = SystemMessage(
        content=(
            "Eres un asistente experto en documentación técnica y gestión de proyectos. "
            "Respondé la pregunta del usuario basándote en el contexto recuperado. "
            "Si el contexto no es suficiente, indicalo claramente. "
            "Respondé siempre en el idioma del usuario.\n\n"
            f"CONTEXTO RECUPERADO:\n{rag_context}"
        )
    )

    response = await llm.ainvoke([sys_msg, HumanMessage(content=user_question)])
    answer = response.content

    return {
        "messages": [AIMessage(content=answer)],
        "final_answer": answer,
    }


# ==============================================================================
# Nodo 5: Respond general (sin RAG — para intenciones "general")
# ==============================================================================

async def respond_general(state: AgentState) -> AgentState:
    """
    Responde a intenciones generales (saludos, preguntas simples)
    sin necesidad de RAG ni Redmine.
    """
    llm = get_llm("nvidia", "openai/gpt-oss-120b", 0.5)

    sys_msg = SystemMessage(
        content=(
            "Eres un asistente amigable de gestión de proyectos. "
            "Respondé de manera concisa y útil. "
            "Respondé siempre en el idioma del usuario."
        )
    )

    response = await llm.ainvoke([sys_msg] + list(state["messages"]))
    answer = response.content

    return {
        "messages": [AIMessage(content=answer)],
        "final_answer": answer,
    }


# ==============================================================================
# Nodo 6: Guardrail de Salida (Guardrails AI)
# ==============================================================================

async def output_guardrail(state: AgentState) -> AgentState:
    """
    Nodo de guardrail de salida usando Guardrails AI.
    Valida la respuesta generada por el LLM/agente antes de entregarla al usuario.
    """
    last_msg = next(
        (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
        None,
    )
    content = str(last_msg.content) if last_msg and last_msg.content else state.get("final_answer", "")

    if not content:
        return state

    # Parsear y validar con Guardrails AI
    guard = Guard()
    try:
        outcome = guard.parse(llm_output=content)
        if outcome.validation_passed:
            validated_text = outcome.validated_output or content
        else:
            validated_text = "Lo siento, la respuesta generada no superó los controles de seguridad y calidad."
    except Exception:
        validated_text = content

    return {
        "messages": [AIMessage(content=validated_text)],
        "final_answer": validated_text,
    }


# ==============================================================================
# Router (Edge Condicional)
# ==============================================================================

def route_after_analyze(state: AgentState) -> str:
    intent = state.get("intent", "general")

    if intent == "search_docs":
        return "rag_query"
    elif intent in ("query_issue", "create_issue", "update_issue", "list_issues"):
        return "redmine_agent"
    else:
        return "respond_general"


# ==============================================================================
# Construcción del Grafo
# ==============================================================================

def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    # ── Nodos ──────────────────────────────────────────────────────────────────
    workflow.add_node("analyze_safe_query", analyze_safe_query)
    workflow.add_node("analyze_intent", analyze_intent)
    workflow.add_node("redmine_agent", make_redmine_agent_node())
    workflow.add_node("rag_query", rag_query)
    workflow.add_node("respond", respond)
    workflow.add_node("respond_general", respond_general)
    workflow.add_node("output_guardrail", output_guardrail)

    # ── Edges ──────────────────────────────────────────────────────────────────
    workflow.add_edge(START, "analyze_safe_query")

    def route_after_safe_query(state: AgentState) -> str:
        if state["is_safe_query"]:
            return "analyze_intent"
        else:
            return "end"

    workflow.add_conditional_edges(
        "analyze_safe_query",
        route_after_safe_query,
        {
            "analyze_intent": "analyze_intent",
            "end": END,
        },
    )

    workflow.add_conditional_edges(
        "analyze_intent",
        route_after_analyze,
        {
            "redmine_agent": "redmine_agent",
            "rag_query": "rag_query",
            "respond_general": "respond_general",
        },
    )

    # Redirigir salidas al Guardrail de Salida
    workflow.add_edge("redmine_agent", "output_guardrail")

    # RAG query → respond → output_guardrail
    workflow.add_edge("rag_query", "respond")
    workflow.add_edge("respond", "output_guardrail")

    # General → output_guardrail
    workflow.add_edge("respond_general", "output_guardrail")

    # El Guardrail de salida finaliza el grafo
    workflow.add_edge("output_guardrail", END)

    return workflow


# ==============================================================================
# Compilación del Grafo
# ==============================================================================

workflow = build_graph()
graph = workflow.compile()