"""
bff/app/chat/router.py
======================
Router de chat con streaming SSE hacia LangGraph Platform.

Endpoints:
  POST /chat/stream          — Inicia streaming SSE del agente.
  POST /chat/threads         — Crea un nuevo thread de conversación.
  GET  /chat/threads/{id}    — Obtiene el historial de un thread.

Protocolo SSE (eventos tipados):
  event: token        → Fragmento de texto del LLM.
  event: tool_call    → Herramienta invocada por el agente.
  event: tool_result  → Resultado de ejecución de herramienta.
  event: node_start   → Inicio de un nodo del grafo.
  event: node_end     → Fin de un nodo del grafo.
  event: error        → Error durante la ejecución.
  event: done         → Fin del stream.
"""

import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from langgraph_sdk import get_client
from pydantic import BaseModel

from app.config import settings
from app.dependencies import get_current_user

router = APIRouter(prefix="/chat", tags=["Chat"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class StreamRequest(BaseModel):
    """Petición de streaming al agente."""
    message: str
    thread_id: str | None = None


class ThreadResponse(BaseModel):
    """Respuesta con el ID del thread creado."""
    thread_id: str


# ---------------------------------------------------------------------------
# Helpers SSE
# ---------------------------------------------------------------------------

def _sse(event: str, data: dict | str) -> str:
    """Formatea un evento SSE según el estándar W3C."""
    payload = json.dumps(data) if isinstance(data, dict) else data
    return f"event: {event}\ndata: {payload}\n\n"


async def _stream_langgraph(
    message: str,
    thread_id: str,
    username: str,
) -> AsyncGenerator[str, None]:
    """
    Conecta al LangGraph Platform, ejecuta el grafo en modo streaming
    y re-emite los eventos como SSE tipados.

    Args:
        message:   Mensaje del usuario.
        thread_id: ID del thread de LangGraph (mantiene el historial).
        username:  Nombre del usuario autenticado (para logs).

    Yields:
        Cadenas de texto SSE formateadas.
    """
    # Construir el cliente asíncrono de LangGraph SDK
    client_kwargs: dict = {"url": settings.langgraph_api_url}
    if settings.langgraph_api_key:
        client_kwargs["api_key"] = settings.langgraph_api_key

    client = get_client(**client_kwargs)

    # Input del grafo — mapea al AgentState definido en graph.py
    graph_input = {
        "messages": [{"role": "user", "content": message}],
        "user_input": message,
    }

    try:
        # Estado para trackear el tamaño de los mensajes y filtrar por nodo
        seen_lengths = {}
        message_nodes = {}
        
        async for chunk in client.runs.stream(
            thread_id,
            settings.langgraph_graph_id,
            input=graph_input,
            stream_mode=["messages", "updates"],
        ):
            event_type: str = chunk.event
            data = chunk.data

            # ── Registro de metadatos (para filtrar por nodo) ───────────────
            if event_type == "messages/metadata":
                if isinstance(data, dict):
                    for msg_id, meta in data.items():
                        node = meta.get("metadata", {}).get("langgraph_node")
                        if node:
                            message_nodes[msg_id] = node

            # ── Streaming de tokens del LLM ─────────────────────────────────
            elif event_type == "messages/partial":
                if isinstance(data, list):
                    for msg in data:
                        msg_id = msg.get("id")
                        if not msg_id:
                            continue
                            
                        # Filtrar mensajes de nodos internos (reasoning/structured output)
                        node = message_nodes.get(msg_id)
                        if node in ("analyze_safe_query", "analyze_intent"):
                            continue
                            
                        content = _extract_content(msg)
                        if content:
                            last_len = seen_lengths.get(msg_id, 0)
                            if len(content) > last_len:
                                delta = content[last_len:]
                                seen_lengths[msg_id] = len(content)
                                yield _sse("token", {"text": delta})

            # ── Actualizaciones de nodos del grafo ──────────────────────────
            elif event_type == "updates":
                if isinstance(data, dict):
                    for node_name, update in data.items():
                        for event in _process_node_update(node_name, update):
                            yield event

            # ── Mensajes completos ──────────────────────────────────────────
            elif event_type == "messages/complete":
                if isinstance(data, list):
                    for msg in data:
                        # Extraer tool calls del mensaje si los hay
                        tool_calls = _extract_tool_calls(msg)
                        for tc in tool_calls:
                            yield _sse("tool_call", tc)

    except Exception as exc:
        yield _sse("error", {"message": str(exc)})
    finally:
        yield _sse("done", {"thread_id": thread_id})


def _extract_content(msg: dict) -> str:
    """Extrae el contenido textual de un mensaje de LangGraph."""
    if isinstance(msg, dict):
        content = msg.get("content", "")
        if isinstance(content, list):
            # Contenido multimodal — concatenar partes de texto
            return "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        return str(content) if content else ""
    return ""


def _extract_tool_calls(msg: dict) -> list[dict]:
    """Extrae los tool_calls de un mensaje AIMessage."""
    if not isinstance(msg, dict):
        return []
    tool_calls = msg.get("tool_calls", [])
    return [
        {
            "name": tc.get("name", ""),
            "args": tc.get("args", {}),
            "id": tc.get("id", ""),
        }
        for tc in tool_calls
        if isinstance(tc, dict)
    ]


def _process_node_update(node_name: str, update: dict) -> list[str]:
    """Genera eventos SSE a partir de la actualización de un nodo del grafo."""
    events: list[str] = []

    if not isinstance(update, dict):
        return events

    # Emitir evento de inicio de nodo para nodos relevantes
    relevant_nodes = {"redmine_agent", "rag_query", "respond", "respond_general", "analyze_intent"}
    if node_name in relevant_nodes:
        events.append(_sse("node_start", {"node": node_name}))

    # Detectar tool calls en los mensajes del update
    for msg in update.get("messages", []):
        if isinstance(msg, dict):
            tool_calls = _extract_tool_calls(msg)
            for tc in tool_calls:
                events.append(_sse("tool_call", {**tc, "node": node_name}))

    # Detectar resultados de herramientas (ToolMessage)
    for msg in update.get("messages", []):
        if isinstance(msg, dict) and msg.get("type") == "tool":
            events.append(_sse("tool_result", {
                "tool_call_id": msg.get("tool_call_id", ""),
                "name": msg.get("name", ""),
                "content": str(msg.get("content", ""))[:500],  # truncar para SSE
            }))

    return events


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/threads",
    response_model=ThreadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Crear un nuevo thread de conversación",
)
async def create_thread(
    current_user: dict = Depends(get_current_user),
) -> ThreadResponse:
    """
    Crea un nuevo thread en LangGraph Platform y retorna su ID.
    El thread_id debe persistirse en el cliente para continuar la conversación.
    """
    client_kwargs: dict = {"url": settings.langgraph_api_url}
    if settings.langgraph_api_key:
        client_kwargs["api_key"] = settings.langgraph_api_key

    client = get_client(**client_kwargs)
    try:
        thread = await client.threads.create()
        return ThreadResponse(thread_id=thread["thread_id"])
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"No se pudo conectar con LangGraph Platform: {exc}",
        )


@router.post(
    "/stream",
    summary="Iniciar streaming SSE del agente",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Stream SSE de eventos tipados del agente",
            "content": {"text/event-stream": {}},
        },
        401: {"description": "Token JWT inválido o expirado"},
    },
)
async def stream_chat(
    body: StreamRequest,
    current_user: dict = Depends(get_current_user),
) -> StreamingResponse:
    """
    Endpoint principal de chat con streaming.

    Si no se provee thread_id, crea uno nuevo automáticamente.
    Los eventos SSE retransmiten en tiempo real:
      - Tokens del LLM (efecto typewriter en el frontend).
      - Tool calls (qué herramienta usa el agente y con qué args).
      - Tool results (resultado de cada herramienta).
      - Eventos de nodo para mostrar el progreso del grafo.
    """
    # Resolver thread_id — crear uno nuevo si no se provee
    thread_id = body.thread_id
    if not thread_id:
        client_kwargs: dict = {"url": settings.langgraph_api_url}
        if settings.langgraph_api_key:
            client_kwargs["api_key"] = settings.langgraph_api_key
        client = get_client(**client_kwargs)
        try:
            thread = await client.threads.create()
            thread_id = thread["thread_id"]
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"No se pudo crear el thread en LangGraph: {exc}",
            )

    username = current_user.get("sub", "unknown")

    return StreamingResponse(
        _stream_langgraph(body.message, thread_id, username),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Desactiva buffering en nginx
            "X-Thread-ID": thread_id,
        },
    )
