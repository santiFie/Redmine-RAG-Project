"""
tests/unit/bff/test_stream_processor.py
======================================
Pruebas unitarias para LangGraphStreamProcessor en el BFF.
Valida el manejo de eventos de error, metadatos y generación de eventos SSE.
"""

import json
import sys
from pathlib import Path

_bff_dir = Path(__file__).resolve().parent.parent.parent.parent / "bff"
sys.path.insert(0, str(_bff_dir))

from app.chat.router import LangGraphStreamProcessor  # noqa: E402


def _parse_sse(raw_event: str) -> tuple[str, dict]:
    """Parsea un evento SSE en (event_type, data_dict)."""
    lines = raw_event.strip().split("\n")
    event_type = ""
    data_dict = {}
    for line in lines:
        if line.startswith("event: "):
            event_type = line[7:].strip()
        elif line.startswith("data: "):
            data_dict = json.loads(line[6:].strip())
    return event_type, data_dict


def test_processor_handles_error_event():
    """Valida que un evento 'error' se procese correctamente emitiendo SSE event error."""
    processor = LangGraphStreamProcessor(thread_id="test-thread-1")
    raw_events = processor.process_chunk("error", {"message": "El modelo falló por quota"})

    assert len(raw_events) == 1
    event_type, data = _parse_sse(raw_events[0])
    assert event_type == "error"
    assert "El modelo falló por quota" in data["message"]
    assert processor.has_error is True
    assert processor.error_emitted is True


def test_processor_handles_metadata_with_error():
    """Valida que un chunk de metadata con status error emita SSE error."""
    processor = LangGraphStreamProcessor(thread_id="test-thread-2")
    raw_events = processor.process_chunk(
        "metadata",
        {"run_id": "run-123", "status": "error", "error": {"message": "Timeout en modelo"}},
    )

    assert len(raw_events) == 1
    event_type, data = _parse_sse(raw_events[0])
    assert event_type == "error"
    assert "Timeout en modelo" in data["message"]
    assert processor.captured_run_id == "run-123"
    assert processor.has_error is True


def test_processor_handles_updates_with_error():
    """Valida que un chunk updates con __error__ emita SSE error."""
    processor = LangGraphStreamProcessor(thread_id="test-thread-3")
    raw_events = processor.process_chunk(
        "updates",
        {"__error__": {"message": "Fallo irrecuperable en nodo"}},
    )

    assert len(raw_events) == 1
    event_type, data = _parse_sse(raw_events[0])
    assert event_type == "error"
    assert "Fallo irrecuperable en nodo" in data["message"]
    assert processor.has_error is True


def test_processor_token_counting():
    """Valida que el conteo de tokens emitidos se actualice correctamente."""
    processor = LangGraphStreamProcessor(thread_id="test-thread-4")
    processor.message_nodes["msg-1"] = "respond"

    raw_events = processor.process_chunk(
        "messages/partial",
        [{"id": "msg-1", "content": "Hola mundo"}],
    )
    assert len(raw_events) == 1
    assert processor.tokens_emitted == len("Hola mundo")
