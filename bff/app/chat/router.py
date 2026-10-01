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
import logging
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from langgraph_sdk import get_client
import langsmith
from pydantic import BaseModel

from app.config import settings
from app.dependencies import get_current_user

router = APIRouter(prefix="/chat", tags=["Chat"])
logger = logging.getLogger("bff.chat")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class StreamRequest(BaseModel):
    """Petición de streaming al agente."""

    message: str
    thread_id: str | None = None


class FeedbackRequest(BaseModel):
    """Payload para enviar feedback a LangSmith."""

    run_id: str
    user_score: float
    text_comment: str | None = None
    value: str | None = None


class ThreadResponse(BaseModel):
    """Respuesta con el ID del thread creado."""

    thread_id: str


# ---------------------------------------------------------------------------
# Helpers SSE y Procesamiento de Stream
# ---------------------------------------------------------------------------


def _sse(event: str, data: dict | str) -> str:
    """Formatea un evento SSE según el estándar W3C."""
    payload = json.dumps(data) if isinstance(data, dict) else data
    return f"event: {event}\ndata: {payload}\n\n"


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


class LangGraphStreamProcessor:
    """
    Encapsula el estado y la lógica de transformación de eventos crudos
    de LangGraph Platform hacia eventos SSE tipados.
    """

    IGNORED_NODES = {"analyze_safe_query", "analyze_intent"}
    RELEVANT_START_NODES = {
        "redmine_agent",
        "rag_query",
        "respond",
        "respond_general",
        "analyze_intent",
    }

    def __init__(self, thread_id: str) -> None:
        self.thread_id = thread_id
        self.seen_lengths: dict[str, int] = {}
        self.message_nodes: dict[str, str] = {}
        self.captured_run_id: str | None = None
        self.has_error: bool = False
        self.error_message: str | None = None
        self.error_emitted: bool = False
        self.tokens_emitted: int = 0

    def process_chunk(self, event_type: str, data: Any) -> list[str]:
        """Despacha el evento al handler correspondiente según el tipo de evento."""
        match event_type:
            case "metadata":
                return self._handle_metadata(data)
            case "messages/metadata":
                return self._handle_messages_metadata(data)
            case "messages/partial":
                return self._handle_messages_partial(data)
            case "updates":
                return self._handle_updates(data)
            case "messages/complete":
                return self._handle_messages_complete(data)
            case "error":
                return self._handle_error(data)
            case _:
                return []

    def _handle_error(self, data: Any) -> list[str]:
        self.has_error = True
        msg = "Error durante la ejecución del agente."
        if isinstance(data, dict):
            msg = data.get("message") or data.get("error") or data.get("detail") or str(data)
        elif data:
            msg = str(data)
        self.error_message = msg
        self.error_emitted = True
        logger.error("Error emitido por LangGraph stream: %s", msg)
        return [_sse("error", {"message": msg})]

    def _handle_metadata(self, data: Any) -> list[str]:
        if not isinstance(data, dict):
            return []
        if "run_id" in data:
            self.captured_run_id = data["run_id"]
        if data.get("status") == "error" or "error" in data:
            err = data.get("error", "Error en la ejecución del grafo")
            msg = (
                (err.get("message") or err.get("error") or str(err))
                if isinstance(err, dict)
                else str(err)
            )
            self.has_error = True
            self.error_message = msg
            self.error_emitted = True
            logger.error("Error detectado en metadata de LangGraph: %s", msg)
            return [_sse("error", {"message": msg})]
        return []

    def _handle_messages_metadata(self, data: Any) -> list[str]:
        if not isinstance(data, dict):
            return []
        for msg_id, meta in data.items():
            if isinstance(meta, dict):
                node = meta.get("metadata", {}).get("langgraph_node")
                if node:
                    self.message_nodes[msg_id] = node
                if "run_id" in meta:
                    self.captured_run_id = meta["run_id"]
        return []

    def _handle_messages_partial(self, data: Any) -> list[str]:
        if not isinstance(data, list):
            return []
        events: list[str] = []
        for msg in data:
            if not isinstance(msg, dict):
                continue
            msg_id = msg.get("id")
            if not msg_id or self.message_nodes.get(msg_id) in self.IGNORED_NODES:
                continue

            content = _extract_content(msg)
            if content:
                last_len = self.seen_lengths.get(msg_id, 0)
                if len(content) > last_len:
                    delta = content[last_len:]
                    self.seen_lengths[msg_id] = len(content)
                    self.tokens_emitted += len(delta)
                    events.append(_sse("token", {"text": delta}))
        return events

    def _handle_updates(self, data: Any) -> list[str]:
        if not isinstance(data, dict):
            return []
        events: list[str] = []
        for node_name, update in data.items():
            if node_name == "__error__" or (isinstance(update, dict) and "error" in update):
                err = update.get("error", update) if isinstance(update, dict) else update
                msg = (
                    (err.get("message") or err.get("error") or str(err))
                    if isinstance(err, dict)
                    else str(err)
                )
                self.has_error = True
                self.error_message = msg
                self.error_emitted = True
                logger.error("Error en update de LangGraph (nodo %s): %s", node_name, msg)
                events.append(_sse("error", {"message": msg}))
            else:
                events.extend(self._process_single_node_update(node_name, update))
        return events

    def _process_single_node_update(self, node_name: str, update: Any) -> list[str]:
        events: list[str] = []
        if not isinstance(update, dict):
            return events

        if node_name in self.RELEVANT_START_NODES:
            events.append(_sse("node_start", {"node": node_name}))

        for msg in update.get("messages", []):
            if not isinstance(msg, dict):
                continue

            # Detectar llamadas a herramientas
            for tc in _extract_tool_calls(msg):
                events.append(_sse("tool_call", {**tc, "node": node_name}))

            # Detectar resultados de herramientas (ToolMessage)
            if msg.get("type") == "tool":
                events.append(
                    _sse(
                        "tool_result",
                        {
                            "tool_call_id": msg.get("tool_call_id", ""),
                            "name": msg.get("name", ""),
                            "content": str(msg.get("content", ""))[:500],  # truncar para SSE
                        },
                    )
                )

        return events

    def _handle_messages_complete(self, data: Any) -> list[str]:
        if not isinstance(data, list):
            return []
        events: list[str] = []
        for msg in data:
            for tc in _extract_tool_calls(msg):
                events.append(_sse("tool_call", tc))
        return events

    def build_done_event(self) -> str:
        """Construye el evento SSE final 'done' con thread_id y run_id."""
        payload: dict[str, str] = {"thread_id": self.thread_id}
        if self.captured_run_id:
            payload["run_id"] = self.captured_run_id
        return _sse("done", payload)


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
    client_kwargs: dict = {"url": settings.langgraph_api_url}
    if settings.langgraph_api_key:
        client_kwargs["api_key"] = settings.langgraph_api_key

    client = get_client(**client_kwargs)

    graph_input = {
        "messages": [{"role": "user", "content": message}],
        "user_input": message,
    }

    processor = LangGraphStreamProcessor(thread_id=thread_id)

    try:
        async for chunk in client.runs.stream(
            thread_id,
            settings.langgraph_graph_id,
            input=graph_input,
            stream_mode=["messages", "updates"],
        ):
            for sse_event in processor.process_chunk(chunk.event, chunk.data):
                yield sse_event

        if processor.has_error and not processor.error_emitted:
            yield _sse(
                "error",
                {"message": processor.error_message or "Error durante la ejecución del agente."},
            )

    except Exception as exc:
        logger.exception("Error durante la ejecución del streaming en LangGraph: %s", exc)
        processor.has_error = True
        processor.error_message = f"Ocurrió un error durante la ejecución del agente: {exc}"
        processor.error_emitted = True
        yield _sse("error", {"message": processor.error_message})
    finally:
        yield processor.build_done_event()


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
        ) from exc


@router.post(
    "/feedback",
    status_code=status.HTTP_201_CREATED,
    summary="Enviar feedback del usuario a LangSmith",
)
async def submit_feedback(
    body: FeedbackRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Envía feedback explícito (pulgar arriba/abajo) a LangSmith
    asociado a un run_id específico.
    """
    try:
        ls_client = langsmith.Client()
        ls_client.create_feedback(
            run_id=body.run_id,
            key="user_score",
            score=body.user_score,
            comment=body.text_comment,
            value=body.value,
            feedback_source_type="api",
        )
        return {"status": "ok"}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al enviar feedback a LangSmith: {exc}",
        ) from exc


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
            ) from exc

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
