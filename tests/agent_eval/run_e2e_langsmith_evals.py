#!/usr/bin/env python3
"""
tests/agent_eval/run_e2e_langsmith_evals.py
===========================================
Suite End-to-End del grafo completo (`src/agent/graph.py`) con LangSmith.

A diferencia de `run_langsmith_evals.py` (nodos aislados), aquí se ejecuta el grafo entero:
  - Se compila con `MemorySaver` SOLO en el runner (el grafo exportado no se toca).
  - Los `interrupt()` reales (clarificación y confirmación) se reanudan con `Command(resume=...)`.
  - `RedmineClient.create_issue` se espía en todos los casos; solo los casos con
    `redmine_mode == "real"` lo ejecutan contra Redmine (se verifica con `get_issue`
    y se elimina con `delete_issue` en el teardown). El resto devuelve un ticket simulado.
  - Las respuestas del "humano" a la clarificación se construyen únicamente con los
    `missing_fields` que el agente realmente solicitó en el `interrupt`.

Uso:
    python -m tests.agent_eval.run_e2e_langsmith_evals --dry-run
    python -m tests.agent_eval.run_e2e_langsmith_evals --sync-datasets
    python -m tests.agent_eval.run_e2e_langsmith_evals --run-evals
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import patch

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASET_FILE = Path(__file__).resolve().parent / "datasets" / "e2e_workflow_ground_truth.json"
DATASET_NAME = "dataset-e2e-workflow"
MAX_RESUMES = 6
FAKE_ISSUE_ID = 99999

KNOWN_NODES = {
    "analyze_safe_query",
    "analyze_intent",
    "rag_knowledge_query",
    "respond_knowledge",
    "qa_evaluator",
    "ask_clarification",
    "duplicate_and_rag_check",
    "confirm_creation",
    "redmine_creator",
    "respond_existing",
    "respond_creation",
    "respond_cancelled",
    "respond_general",
    "output_guardrail",
}


# ==============================================================================
# Utilidades de simulación del humano (HITL)
# ==============================================================================


def build_clarification_answer(payload: dict[str, Any], human: dict[str, Any]) -> str:
    """
    Arma la respuesta del usuario SOLO con los campos que el agente pidió.
    Los campos pedidos sin respuesta preparada reciben la respuesta de fallback
    ('no sé'), que se agrega una única vez.
    """
    asked = [str(f).lower() for f in payload.get("missing_fields", [])]
    rules = human.get("clarification_field_answers", [])
    fallback = human.get("clarification_fallback", "No sé.")

    answers: list[str] = []
    used: set[int] = set()
    unmatched = False
    for field in asked:
        idx = next(
            (
                i
                for i, r in enumerate(rules)
                if any(k.lower() in field for k in r.get("keywords", []))
            ),
            None,
        )
        if idx is None:
            unmatched = True
        elif idx not in used:
            used.add(idx)
            answers.append(rules[idx]["answer"])
    if unmatched or not asked:
        answers.append(fallback)
    return " ".join(answers)


def build_resume_value(payload: dict[str, Any], human: dict[str, Any]) -> Any:
    action = payload.get("action")
    if action == "provide_clarification":
        return build_clarification_answer(payload, human)
    if action == "confirm_issue_creation":
        return human.get("confirmation", "confirmar")
    raise RuntimeError(f"Interrupt inesperado: {payload}")


# ==============================================================================
# Target: ejecuta el grafo completo
# ==============================================================================


async def run_workflow(inputs: dict[str, Any]) -> dict[str, Any]:
    """Run function de LangSmith. Debe ejecutarse con max_concurrency=1 (patch global)."""
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    from src.agent.graph import build_graph
    from src.redmine.client import RedmineClient

    mode = inputs.get("redmine_mode", "mock")
    human = inputs.get("human_responses", {})
    calls: list[dict[str, Any]] = []
    real_original = RedmineClient.create_issue
    created_real: dict[str, Any] = {}

    def spy_create_issue(self: RedmineClient, *args: Any, **kwargs: Any) -> dict[str, Any]:
        record = dict(kwargs)
        if args:
            record.update(zip(("project_id", "subject", "description"), args))
        calls.append(record)
        if mode == "real":
            issue = real_original(self, *args, **kwargs)
            created_real["id"] = issue["id"]
            created_real["record"] = record
            return issue
        return {"id": FAKE_ISSUE_ID}

    graph = build_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": f"e2e-{uuid.uuid4()}"}}

    trajectory: list[str] = []
    interrupts: list[dict[str, Any]] = []
    next_input: Any = {"user_input": inputs["user_input"], "messages": []}
    values: dict[str, Any] = {}
    verified = False

    try:
        with patch.object(RedmineClient, "create_issue", spy_create_issue):
            for _ in range(MAX_RESUMES + 1):
                async for chunk in graph.astream(next_input, config, stream_mode="updates"):
                    for node in chunk:
                        if node != "__interrupt__":
                            trajectory.append(node)

                snapshot = await graph.aget_state(config)
                values = snapshot.values
                pending = [i for t in snapshot.tasks for i in t.interrupts]
                if not pending:
                    break
                payload = pending[0].value
                interrupts.append(payload)
                next_input = Command(resume=build_resume_value(payload, human))
            else:
                raise RuntimeError("Se superó el máximo de reanudaciones HITL (posible bucle).")

        if mode == "real" and created_real:
            verified = await _verify_real_issue(created_real)
    finally:
        if mode == "real" and created_real.get("id"):
            await _delete_real_issue(created_real["id"])

    analysis = values.get("bug_analysis")
    return {
        "trajectory": trajectory,
        "final_answer": values.get("final_answer", ""),
        "intent": values.get("intent"),
        "is_safe_query": values.get("is_safe_query"),
        "is_duplicate": values.get("is_duplicate"),
        "duplicate_issue_id": values.get("duplicate_issue_id"),
        "retrieved_issue_ids": values.get("retrieved_issue_ids", []),
        "rag_context_found": bool(values.get("rag_context")),
        "target_project_id": values.get("target_project_id"),
        "creation_cancelled": values.get("creation_cancelled", False),
        "create_issue_calls": calls,
        "created_issue_id": values.get("created_issue_id"),
        "verified_in_redmine": verified,
        "interrupts": interrupts,
        "bug_analysis": analysis.model_dump() if hasattr(analysis, "model_dump") else analysis,
    }


async def _verify_real_issue(created: dict[str, Any]) -> bool:
    from src.redmine.client import RedmineClient

    def _check() -> bool:
        with RedmineClient() as client:
            issue = client.get_issue(created["id"])
        return issue.get("subject") == created["record"].get("subject")

    try:
        return await asyncio.to_thread(_check)
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️ No se pudo verificar el ticket real #{created['id']}: {e}")
        return False


async def _delete_real_issue(issue_id: int) -> None:
    from src.redmine.client import RedmineClient

    def _delete() -> None:
        with RedmineClient() as client:
            client.delete_issue(issue_id)

    try:
        await asyncio.to_thread(_delete)
        print(f"  🧹 Ticket de prueba #{issue_id} eliminado de Redmine.")
    except Exception as e:  # noqa: BLE001
        print(f"  ❌ No se pudo eliminar el ticket de prueba #{issue_id}: {e}")


# ==============================================================================
# Datasets / ejecución
# ==============================================================================


def load_examples() -> list[dict[str, Any]]:
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        return json.load(f)["examples"]


def dry_run() -> None:
    examples = load_examples()
    errors: list[str] = []
    for ex in examples:
        exp, inp = ex["outputs"], ex["inputs"]
        unknown = set(exp.get("expected_trajectory", [])) - KNOWN_NODES
        if unknown:
            errors.append(f"{ex['id']}: nodos desconocidos {sorted(unknown)}")
        if "user_input" not in inp:
            errors.append(f"{ex['id']}: falta user_input")
        if inp.get("redmine_mode") != exp.get("redmine_mode"):
            errors.append(f"{ex['id']}: redmine_mode inconsistente")
        if exp.get("expected_issue_created") and not exp.get("expected_target_project"):
            errors.append(f"{ex['id']}: falta expected_target_project")
    reals = [e["id"] for e in examples if e["inputs"].get("redmine_mode") == "real"]
    if len(reals) != 1:
        errors.append(f"Debe haber exactamente 1 caso 'real' (hay {len(reals)}: {reals})")
    for ex in examples:
        print(f"  ✅ {ex['id']}: {ex['description'][:90]}")
    if errors:
        print("\n❌ Errores:\n  " + "\n  ".join(errors))
        raise SystemExit(1)
    print(f"\n✨ Dry-run OK: {len(examples)} casos, 1 real ({reals[0]}).")


def sync_dataset() -> None:
    from langsmith import Client
    from langsmith.schemas import DataType

    client = Client()
    examples = load_examples()
    if client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
    else:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="Dataset E2E del grafo completo (trayectoria, HITL, side-effects, respuesta final).",
            data_type=DataType.kv,
        )
    existing = {
        json.dumps(e.inputs, sort_keys=True) for e in client.list_examples(dataset_id=dataset.id)
    }
    created = 0
    for ex in examples:
        if json.dumps(ex["inputs"], sort_keys=True) in existing:
            continue
        client.create_example(
            inputs=ex["inputs"],
            outputs=ex["outputs"],
            metadata={"id": ex["id"], "description": ex["description"], **ex.get("metadata", {})},
            dataset_id=dataset.id,
        )
        created += 1
    print(f"✅ Dataset '{DATASET_NAME}': {created} ejemplos nuevos (total {len(examples)}).")


async def run_evals() -> None:
    from langsmith.evaluation import aevaluate

    from src.evaluation.e2e_evaluators import (
        e2e_clarification_alignment_evaluator,
        e2e_disclosure_completeness_evaluator,
        e2e_final_contract_evaluator,
        e2e_side_effects_evaluator,
        e2e_trajectory_evaluator,
    )

    await aevaluate(
        run_workflow,
        data=DATASET_NAME,
        evaluators=[
            e2e_trajectory_evaluator,
            e2e_side_effects_evaluator,
            e2e_clarification_alignment_evaluator,
            e2e_final_contract_evaluator,
            e2e_disclosure_completeness_evaluator,
        ],
        experiment_prefix="eval-e2e-workflow",
        max_concurrency=1,  # create_issue se parchea globalmente -> ejecución secuencial
    )
    print("🎉 Experimento E2E finalizado.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluaciones E2E del grafo en LangSmith.")
    parser.add_argument("--dry-run", action="store_true", help="Valida el dataset local sin red.")
    parser.add_argument("--sync-datasets", action="store_true", help="Sube el dataset a LangSmith.")
    parser.add_argument("--run-evals", action="store_true", help="Ejecuta el experimento E2E.")
    args = parser.parse_args()

    if not (args.dry_run or args.sync_datasets or args.run_evals):
        parser.print_help()
        return
    if args.dry_run:
        dry_run()
        return
    if args.sync_datasets:
        sync_dataset()
    if args.run_evals:
        asyncio.run(run_evals())


if __name__ == "__main__":
    main()
