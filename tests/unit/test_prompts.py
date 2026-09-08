"""
tests/unit/test_prompts.py
==========================
Valida la carga, estructura e instanciación de todas las plantillas de prompts en src/prompts.
"""

import pytest
from langchain_core.prompts import ChatPromptTemplate
from src.prompts.loader import aget_prompt, aload_all_prompts, get_prompt, load_all_prompts


def test_load_all_prompts_contains_expected_keys():
    """Verifica que todas las plantillas esenciales existan y se carguen."""
    prompts = load_all_prompts()

    expected_prompts = [
        "analyze-safe-query",
        "analyze-intent",
        "redmine-agent",
        "respond-rag",
        "respond-general",
        "qa-evaluator",
        "ask-clarification",
        "triage-duplicate-check",
    ]

    for name in expected_prompts:
        assert name in prompts, f"El prompt '{name}' no fue encontrado en las plantillas cargadas."
        template = prompts[name]
        assert isinstance(template, ChatPromptTemplate)


def test_triage_duplicate_check_variables():
    """Valida que triage-duplicate-check tenga las variables enriched_query, rag_context y projects_text."""
    prompt = get_prompt("triage-duplicate-check")
    assert isinstance(prompt, ChatPromptTemplate)

    input_vars = set(prompt.input_variables)
    expected_vars = {"enriched_query", "rag_context", "projects_text"}
    assert expected_vars.issubset(input_vars), (
        f"Variables esperadas: {expected_vars}, obtenidas: {input_vars}"
    )

    # Formatear el prompt con valores dummy
    messages = prompt.format_messages(
        enriched_query="Error 500 al guardar factura",
        rag_context="Ticket #102: Error al guardar factura en v1.2",
        projects_text="- ID: 'facturacion' | Nombre: 'Facturación'",
    )
    assert len(messages) == 2

    # Contenido del mensaje del sistema (projects_text)
    system_content = messages[0].content
    assert "facturacion" in system_content

    # Contenido del mensaje humano (enriched_query y rag_context)
    human_content = messages[1].content
    assert "Error 500 al guardar factura" in human_content
    assert "Ticket #102" in human_content


def test_qa_evaluator_variables():
    """Valida que qa-evaluator acepte question y genere mensajes formateados."""
    prompt = get_prompt("qa-evaluator")
    assert "question" in prompt.input_variables

    messages = prompt.format_messages(question="No puedo iniciar sesión en el CRM")
    assert len(messages) == 2  # System + Human
    assert "No puedo iniciar sesión en el CRM" in messages[1].content


def test_ask_clarification_variables():
    """Valida que ask-clarification acepte missing_fields."""
    prompt = get_prompt("ask-clarification")
    assert "missing_fields" in prompt.input_variables

    messages = prompt.format_messages(missing_fields="['pasos para reproducir', 'navegador']")
    assert "pasos para reproducir" in messages[0].content


def test_get_prompt_key_error_on_unknown():
    """Falla apropiadamente con KeyError si el prompt no existe."""
    with pytest.raises(KeyError):
        get_prompt("prompt-inexistente-12345")


@pytest.mark.asyncio
async def test_aget_prompt_and_aload_all_prompts():
    """Valida la carga asíncrona de prompts con aget_prompt y aload_all_prompts."""
    prompts = await aload_all_prompts()
    assert "analyze-intent" in prompts

    prompt = await aget_prompt("analyze-intent")
    assert isinstance(prompt, ChatPromptTemplate)
    assert "user_input" in prompt.input_variables

    with pytest.raises(KeyError):
        await aget_prompt("prompt-inexistente-12345")
