#!/usr/bin/env python3
"""
tests/agent_eval/run_langsmith_evals.py
=======================================
Script CLI para sincronizar datasets curados con LangSmith, configurar la cola
de anotación humana (Annotation Queue para HITL) y ejecutar las evaluaciones
de los nodos LLM del agente mediante Decision Models de OpenRouter (typesafe/jev-1.13).

Uso:
    # 1. Cargar/Actualizar datasets y configurar colas en LangSmith
    python -m tests.agent_eval.run_langsmith_evals --sync-datasets

    # 2. Ejecutar suite de evaluación completa con Decision Models
    python -m tests.agent_eval.run_langsmith_evals --run-evals

    # 3. Todo junto
    python -m tests.agent_eval.run_langsmith_evals --sync-datasets --run-evals
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

# Cargar variables de entorno del proyecto
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langsmith import Client
from langsmith.evaluation import aevaluate
from langsmith.schemas import DataType, Example, Run

from src.evaluation.decision_models_evaluators import (
    DECISION_EVAL_MODEL,
    OPENROUTER_BASE_URL,
    clarification_coverage_evaluator,
    clarification_no_out_of_scope_evaluator,
    clarification_no_redundancy_evaluator,
    clarification_relevance_evaluator,
    clarification_rule_compliance_evaluator,
    clarification_tone_clarity_evaluator,
    duplicate_detection_evaluator,
    extraction_completeness_evaluator,
    get_decision_evaluator_classifier,
    incident_type_evaluator,
    no_hallucination_evaluator,
    project_routing_evaluator,
    sufficiency_decision_evaluator,
    triage_reasoning_evaluator,
)
from src.evaluation.llm_judge_evaluators import sync_judge_prompt_to_hub
from tests.agent_eval.audit_llm_judge import audit_flagged_cases

DATASETS_DIR = Path(__file__).resolve().parent / "datasets"

DATASET_CONFIGS: list[dict[str, Any]] = [
    {
        "file": DATASETS_DIR / "qa_extraction_ground_truth.json",
        "dataset_name": "dataset-qa-extraction",
        "data_type": DataType.kv,
        "description": "Dataset para evaluar la extracción estructurada del nodo qa_evaluator_node hacia BugReportExtraction.",
    },
    {
        "file": DATASETS_DIR / "ask_clarification_ground_truth.json",
        "dataset_name": "dataset-ask-clarification",
        "data_type": DataType.kv,
        "description": "Dataset para evaluar la pertinencia y empatía de las preguntas generadas en ask_clarification_node.",
    },
    {
        "file": DATASETS_DIR / "duplicate_and_rag_ground_truth.json",
        "dataset_name": "dataset-triage-duplicate-check",
        "data_type": DataType.kv,
        "description": "Dataset para evaluar la deduplicación y el enrutamiento a proyectos en duplicate_and_rag_check_node.",
    },
]

ANNOTATION_QUEUE_NAME = os.getenv("LANGSMITH_ANNOTATION_QUEUE", "redmine-qa-human-review")


def get_langsmith_client() -> Client:
    """Instancia el cliente de LangSmith verificando credenciales."""
    api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        print("⚠️  LANGCHAIN_API_KEY / LANGSMITH_API_KEY no encontrada en el entorno.")
    return Client(api_key=api_key)


def sync_datasets_to_langsmith(client: Client) -> dict[str, Any]:
    """Carga o actualiza los datasets en LangSmith a partir de los archivos JSON locales."""
    print("=" * 80)
    print("📦 SINCRONIZANDO DATASETS GROUND TRUTH CON LANGSMITH")
    print("=" * 80)

    results: dict[str, Any] = {}
    for cfg in DATASET_CONFIGS:
        file_path: Path = cfg["file"]
        dataset_name: str = cfg["dataset_name"]
        description: str = cfg["description"]
        raw_type = cfg["data_type"]
        data_type: DataType = (
            raw_type if isinstance(raw_type, DataType) else DataType(str(raw_type))
        )

        if not file_path.exists():
            print(f"❌ Archivo no encontrado: {file_path}")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        examples = data.get("examples", [])
        print(f"\nProcesando '{dataset_name}' ({len(examples)} ejemplos desde {file_path.name})...")

        # Verificar si el dataset ya existe en LangSmith
        dataset = None
        try:
            if client.has_dataset(dataset_name=dataset_name):
                dataset = client.read_dataset(dataset_name=dataset_name)
                print(f"  ℹ️ Dataset existente encontrado (ID: {dataset.id}).")
            else:
                dataset = client.create_dataset(
                    dataset_name=dataset_name,
                    description=description,
                    data_type=data_type,
                )
                print(f"  ✅ Dataset creado exitosamente (ID: {dataset.id}).")
        except Exception as e:
            print(f"  ❌ Error al conectar con LangSmith para dataset '{dataset_name}': {e}")
            continue

        # Sincronizar ejemplos existentes para evitar duplicados
        try:
            existing_examples = list(client.list_examples(dataset_id=dataset.id))
            existing_inputs = {
                json.dumps(ex.inputs, sort_keys=True): ex.id for ex in existing_examples
            }

            created_count = 0
            for ex in examples:
                inp = ex["inputs"]
                out = ex.get("outputs", {})
                meta = ex.get("metadata", {})

                inp_key = json.dumps(inp, sort_keys=True)
                if inp_key not in existing_inputs:
                    client.create_example(
                        inputs=inp,
                        outputs=out,
                        metadata=meta,
                        dataset_id=dataset.id,
                    )
                    created_count += 1

            print(f"  ✨ Ejemplos cargados: {created_count} nuevos (Total: {len(examples)}).")
            results[dataset_name] = {"id": str(dataset.id), "total_examples": len(examples)}
        except Exception as e:
            print(f"  ❌ Error al sincronizar ejemplos en dataset '{dataset_name}': {e}")

    return results


def setup_annotation_queue(client: Client) -> str | None:
    """Crea o asegura la existencia de la Annotation Queue para revisión humana (HITL)."""
    print("\n" + "=" * 80)
    print("👤 CONFIGURANDO ANNOTATION QUEUE (HUMAN-IN-THE-LOOP)")
    print("=" * 80)

    try:
        queues = list(client.list_annotation_queues())
        for q in queues:
            if getattr(q, "name", None) == ANNOTATION_QUEUE_NAME:
                print(f"ℹ️ Annotation Queue '{ANNOTATION_QUEUE_NAME}' ya existe (ID: {q.id}).")
                return str(q.id)

        queue = client.create_annotation_queue(
            name=ANNOTATION_QUEUE_NAME,
            description="Cola de auditoría humana para revisar decisiones de triaje, extracción y evaluadores de OpenRouter.",
        )
        print(f"✅ Annotation Queue '{ANNOTATION_QUEUE_NAME}' creada con éxito (ID: {queue.id}).")
        return str(queue.id)
    except Exception as e:
        print(f"⚠️ Advertencia al gestionar Annotation Queue: {e}")
        return None


def sync_prompts_to_langsmith(client: Client) -> None:
    """Publica o actualiza las rúbricas y prompts de evaluación en LangSmith Prompt Hub."""
    print("\n" + "=" * 80)
    print("📝 SINCRONIZANDO PROMPTS CON LANGSMITH PROMPT HUB")
    print("=" * 80)
    try:
        url = sync_judge_prompt_to_hub(client)
        print(f"  ✅ Prompt 'clarification-judge-rubric' sincronizado con éxito: {url}")
    except Exception as e:
        print(f"  ⚠️ No se pudo sincronizar el prompt con LangSmith Hub: {e}")


# ==============================================================================
# Predictors aislados para los Nodos LLM del Agente
# ==============================================================================


async def target_qa_evaluator_node(inputs: dict[str, Any]) -> dict[str, Any]:
    """Ejecutor predictivo para qa_evaluator_node."""
    from src.agent.graph import qa_evaluator_node
    from src.agent.state import State

    user_input = inputs.get("user_input", "")
    state: State = {
        "messages": [],
        "user_input": user_input,
        "is_safe_query": True,
        "intent": "incident_report",
        "final_answer": "",
    }
    result = await qa_evaluator_node(state)
    analysis = result.get("bug_analysis")
    if analysis:
        # Serializar BugReportExtraction a dict
        return analysis.model_dump() if hasattr(analysis, "model_dump") else dict(analysis)
    return {}


async def target_ask_clarification_node(inputs: dict[str, Any]) -> dict[str, Any]:
    """Ejecutor predictivo para ask_clarification_node."""
    from src.agent.graph import ask_clarification_node
    from src.agent.state import BugReportExtraction, State
    from unittest.mock import patch

    user_input = inputs.get("user_input", "")
    missing_fields = inputs.get("missing_fields", [])

    state: State = {
        "messages": [],
        "user_input": user_input,
        "is_safe_query": True,
        "intent": "incident_report",
        "bug_analysis": BugReportExtraction(
            is_sufficient=False,
            title_summary="Incidente incompleto de prueba",
            missing_fields=missing_fields,
        ),
        "clarification_turns": 0,
        "final_answer": "",
    }

    # Interceptar interrupt para obtener la pregunta generada sin detener la ejecución de evaluación
    with patch("src.agent.graph.interrupt") as mock_interrupt:
        mock_interrupt.return_value = "Respuesta simulada para evals"
        await ask_clarification_node(state)
        interrupt_args = mock_interrupt.call_args[0][0] if mock_interrupt.call_args else {}
        question = interrupt_args.get("question", "")

    return {
        "question": question,
        "missing_fields": missing_fields,
    }


async def target_duplicate_and_rag_check_node(inputs: dict[str, Any]) -> dict[str, Any]:
    """Ejecutor predictivo para duplicate_and_rag_check_node."""
    from src.agent.config import agent_llm_config
    from src.agent.graph import DuplicateCheckResult
    from src.prompts.loader import aget_prompt
    from src.utils.get_llm import get_llm

    enriched_query = inputs.get("enriched_query", "")
    rag_context = inputs.get("rag_context", "")
    projects_text = inputs.get("projects_text", "")

    cfg = agent_llm_config.duplicate_check
    llm = get_llm(cfg.provider, cfg.model, cfg.temperature)
    structured_llm = llm.with_structured_output(DuplicateCheckResult)

    prompt_template = await aget_prompt("triage-duplicate-check")
    messages = prompt_template.format_messages(
        enriched_query=enriched_query,
        rag_context=rag_context,
        projects_text=projects_text,
    )
    result = await structured_llm.ainvoke(messages)
    return result.model_dump() if hasattr(result, "model_dump") else dict(result)


# ==============================================================================
# Ejecución de Evaluaciones con LangSmith y Decision Models
# ==============================================================================


async def run_langsmith_evaluations(client: Client):
    """Ejecuta los experimentos de evaluación reportando scores en LangSmith."""
    print("\n" + "=" * 80)
    print(f"🧪 INICIANDO EVALUACIONES EN LANGSMITH CON DECISION MODEL: {DECISION_EVAL_MODEL}")
    print("=" * 80)

    # 1. Evaluación del Nodo QA Evaluator (Extracción, Completitud, No Alucinación, Tipo, Suficiencia)
    print("\n[1/3] Evaluando 'qa_evaluator_node' contra 'dataset-qa-extraction'...")
    try:
        await aevaluate(
            target_qa_evaluator_node,
            data="dataset-qa-extraction",
            evaluators=[
                sufficiency_decision_evaluator,
                extraction_completeness_evaluator,
                no_hallucination_evaluator,
                incident_type_evaluator,
            ],
            experiment_prefix="eval-qa-extraction-jev",
            client=client,
        )
        print("  ✅ Experimento QA Evaluator completado.")
    except Exception as e:
        print(f"  ❌ Error durante la evaluación de QA Evaluator: {e}")

    # 2. Evaluación del Nodo Ask Clarification (5 Dimensiones con Decision Models)
    print("\n[2/3] Evaluando 'ask_clarification_node' contra 'dataset-ask-clarification'...")
    try:
        eval_results = await aevaluate(
            target_ask_clarification_node,
            data="dataset-ask-clarification",
            evaluators=[
                clarification_coverage_evaluator,
                clarification_no_out_of_scope_evaluator,
                clarification_no_redundancy_evaluator,
                clarification_rule_compliance_evaluator,
                clarification_tone_clarity_evaluator,
            ],
            experiment_prefix="eval-clarification-jev",
            client=client,
        )
        print("  ✅ Experimento Ask Clarification completado.")

        # ----------------------------------------------------------------------
        # Capa 3: Verificación condicional para LLM-as-a-Judge (Score < 0.60)
        # NUNCA se ejecuta de forma automática sin confirmación explícita del usuario
        # ----------------------------------------------------------------------
        flagged_cases = []
        async for row in eval_results:
            run_obj = row.get("run")
            example_obj = row.get("example")
            eval_results_obj = row.get("evaluation_results")

            min_score = 1.0
            worst_comment = ""
            results_list = (
                eval_results_obj.get("results", [])
                if isinstance(eval_results_obj, dict)
                else getattr(eval_results_obj, "results", [])
            )
            for r in results_list:
                s = getattr(r, "score", 1.0) if not isinstance(r, dict) else r.get("score", 1.0)
                if s is not None and s < min_score:
                    min_score = s
                    worst_comment = (
                        getattr(r, "comment", "")
                        if not isinstance(r, dict)
                        else r.get("comment", "")
                    )

            if min_score < 0.60:
                run_outputs = getattr(run_obj, "outputs", {}) or {}
                flagged_cases.append(
                    {
                        "id": getattr(example_obj, "id", "desconocido"),
                        "run_id": getattr(run_obj, "id", None),
                        "inputs": getattr(example_obj, "inputs", {}) if example_obj else {},
                        "question_text": run_outputs.get("question", "")
                        if isinstance(run_outputs, dict)
                        else str(run_outputs),
                        "expected": getattr(example_obj, "outputs", {}) if example_obj else {},
                        "decision_model_score": min_score,
                        "comment": worst_comment,
                    }
                )

        if flagged_cases:
            print("\n======================================================================")
            print(
                f"⚠️  ALERTA DE CALIDAD: Se detectaron {len(flagged_cases)} casos con score < 0.60 en Decision Models:"
            )
            for c in flagged_cases:
                print(
                    f"   • Caso {c['id']} (Run: {c['run_id']}): Score={c['decision_model_score']} -> {c['comment']}"
                )
            print("======================================================================")

            try:
                confirm = (
                    input(
                        "\n¿Desea ejecutar la auditoría profunda con LLM-as-a-Judge sobre estos casos? [s/N]: "
                    )
                    .strip()
                    .lower()
                )
            except (EOFError, KeyboardInterrupt):
                confirm = "n"

            if confirm in ("s", "si", "y", "yes"):
                await audit_flagged_cases(flagged_cases, client=client)
            else:
                print(
                    "ℹ️ Auditoría con LLM-as-a-Judge omitida (no autorizada explícitamente por el usuario)."
                )

    except Exception as e:
        print(f"  ❌ Error durante la evaluación de Ask Clarification: {e}")

    # 3. Evaluación del Nodo Duplicate & Triage Check
    print(
        "\n[3/3] Evaluando 'duplicate_and_rag_check_node' contra 'dataset-triage-duplicate-check'..."
    )
    try:
        await aevaluate(
            target_duplicate_and_rag_check_node,
            data="dataset-triage-duplicate-check",
            evaluators=[
                duplicate_detection_evaluator,
                project_routing_evaluator,
                triage_reasoning_evaluator,
            ],
            experiment_prefix="eval-triage-dedup",
            client=client,
        )
        print("  ✅ Experimento Triage Duplicate Check completado.")
    except Exception as e:
        print(f"  ❌ Error durante la evaluación de Triage Duplicate Check: {e}")

    print("\n🎉 Proceso de evaluación en LangSmith finalizado.")


def main():
    parser = argparse.ArgumentParser(
        description="Gestor de Datasets y Evaluaciones de LangSmith con Decision Models de OpenRouter."
    )
    parser.add_argument(
        "--sync-datasets",
        action="store_true",
        help="Sincroniza y sube los datasets JSON locales a LangSmith.",
    )
    parser.add_argument(
        "--sync-prompts",
        action="store_true",
        help="Sincroniza y sube los prompts de evaluación al Prompt Hub de LangSmith.",
    )
    parser.add_argument(
        "--run-evals",
        action="store_true",
        help="Lanza la suite de evaluación con langsmith.evaluate y Decision Models.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verifica los archivos locales de dataset y evaluadores sin contactar a LangSmith ni OpenRouter.",
    )

    args = parser.parse_args()

    # Si no se pasan argumentos, ejecutar modo ayuda o dry-run informativo
    if not (args.sync_datasets or args.sync_prompts or args.run_evals or args.dry_run):
        parser.print_help()
        print("\nEjemplo: python -m tests.agent_eval.run_langsmith_evals --sync-datasets")
        return

    if args.dry_run:
        print("🔍 Verificando integridad de datasets locales y evaluadores...")
        for cfg in DATASET_CONFIGS:
            with open(cfg["file"], "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"  ✅ {cfg['dataset_name']}: {len(data.get('examples', []))} casos validados.")
        print(f"  ✅ Modelo de decisión configurado: {DECISION_EVAL_MODEL}")
        print("✨ Verificación en seco completada sin errores.")
        return

    client = get_langsmith_client()

    if args.sync_prompts or args.sync_datasets:
        sync_prompts_to_langsmith(client)

    if args.sync_datasets:
        sync_datasets_to_langsmith(client)
        setup_annotation_queue(client)

    if args.run_evals:
        asyncio.run(run_langsmith_evaluations(client))


if __name__ == "__main__":
    main()
