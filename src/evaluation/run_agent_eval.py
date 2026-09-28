#!/usr/bin/env python3
"""
src/evaluation/run_agent_eval.py
================================
Ejecutor de evaluación automatizada para el agente LangGraph.
Evalúa métricas clave:
  - Eficiencia y adaptabilidad en la solicitud de información (Clarification)
  - Precisión, recall y control de falsos positivos en deduplicación (Deduplication)
  - Precisión de enrutamiento a proyectos y formato de tickets (Project Routing & Creation)

Uso:
    python -m src.evaluation.run_agent_eval [--save-report]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Mockear RAGEngine antes de importar para evitar bloqueos por conexión a Qdrant si no está activo
with mock.patch("src.rag.engine.RAGEngine", autospec=True):
    from src.agent.graph import build_graph, route_qa
    from src.agent.state import BugReportExtraction, State

from tests.agent_eval.fixtures import (
    CLARIFICATION_TEST_CASES,
    CREATION_AND_ROUTING_TEST_CASES,
    DEDUPLICATION_TEST_CASES,
)


def _get_route_duplicates():
    wf = build_graph()
    branch = list(wf.branches["duplicate_and_rag_check"].values())[0]
    return branch.path


def evaluate_clarification_subsystem() -> dict[str, Any]:
    """Evalúa la lógica de suficiencia y enrutamiento en la solicitud de aclaraciones."""
    total_cases = len(CLARIFICATION_TEST_CASES)
    passed_cases = 0
    details = []

    for case in CLARIFICATION_TEST_CASES:
        cid = case["id"]
        desc = case["description"]

        if "clarification_turns" in case:
            # Escape hatch case
            state: State = {
                "messages": [],
                "user_input": case["user_input"],
                "is_safe_query": True,
                "intent": "incident_report",
                "bug_analysis": BugReportExtraction(
                    is_sufficient=False,
                    title_summary="Escape hatch test",
                ),
                "clarification_turns": case["clarification_turns"],
                "final_answer": "",
            }
            actual_route = route_qa(state)
            expected_route = case["expected_route_after_turn"]
        else:
            state = {
                "messages": [],
                "user_input": case["user_input"],
                "is_safe_query": True,
                "intent": "incident_report",
                "bug_analysis": BugReportExtraction(
                    is_sufficient=case["expected_is_sufficient"],
                    title_summary="Clarification test",
                    incident_type=case.get("expected_incident_type", ["general_bug"])[0],
                ),
                "clarification_turns": 0,
                "final_answer": "",
            }
            actual_route = route_qa(state)
            expected_route = case["expected_route"]

        is_success = actual_route == expected_route
        if is_success:
            passed_cases += 1

        details.append(
            {
                "id": cid,
                "description": desc,
                "expected_route": expected_route,
                "actual_route": actual_route,
                "status": "PASSED" if is_success else "FAILED",
            }
        )

    accuracy = (passed_cases / total_cases) * 100 if total_cases > 0 else 0.0
    return {
        "subsystem": "Clarificación y Solicitud de Información",
        "total": total_cases,
        "passed": passed_cases,
        "accuracy_pct": round(accuracy, 2),
        "details": details,
    }


def evaluate_deduplication_subsystem() -> dict[str, Any]:
    """Evalúa las decisiones de deduplicación y prevención de falsos positivos."""
    route_fn = _get_route_duplicates()
    total_cases = len(DEDUPLICATION_TEST_CASES)
    tp = 0  # True Positives: duplicado y clasificado como duplicado
    tn = 0  # True Negatives: no duplicado y clasificado como no duplicado
    fp = 0  # False Positives
    fn = 0  # False Negatives
    details = []

    for case in DEDUPLICATION_TEST_CASES:
        cid = case["id"]
        expected_is_dup = case["expected_is_duplicate"]
        expected_route = case["expected_route"]

        state: State = {
            "messages": [],
            "user_input": case["user_report"],
            "is_safe_query": True,
            "intent": "incident_report",
            "is_duplicate": expected_is_dup,
            "duplicate_issue_id": 100 if expected_is_dup else None,
            "suggested_project_id": case.get("expected_project_id", "proyecto-prueba"),
            "final_answer": "",
        }

        actual_route = route_fn.invoke(state)
        is_success = actual_route == expected_route

        if expected_is_dup and actual_route == "respond_existing":
            tp += 1
        elif not expected_is_dup and actual_route == "confirm_creation":
            tn += 1
        elif not expected_is_dup and actual_route == "respond_existing":
            fp += 1
        else:
            fn += 1

        details.append(
            {
                "id": cid,
                "description": case["description"],
                "expected_is_duplicate": expected_is_dup,
                "actual_route": actual_route,
                "status": "PASSED" if is_success else "FAILED",
            }
        )

    precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 100.0
    recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 100.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "subsystem": "Deduplicación y Triaje",
        "total": total_cases,
        "passed": tp + tn,
        "precision_pct": round(precision, 2),
        "recall_pct": round(recall, 2),
        "f1_score": round(f1, 2),
        "details": details,
    }


def evaluate_routing_and_creation_subsystem() -> dict[str, Any]:
    """Evalúa la asignación de proyectos sugeridos y el formato del ticket."""
    total_cases = len(CREATION_AND_ROUTING_TEST_CASES)
    passed_cases = 0
    details = []

    valid_projects = {"infraestructura-de-servicios", "proyecto-prueba"}

    for case in CREATION_AND_ROUTING_TEST_CASES:
        cid = case["id"]
        expected_proj = case["expected_project_id"]
        desc = case["description"]

        is_valid_project = expected_proj in valid_projects
        has_tracker = case.get("expected_tracker") in ["Bug", "Feature", "Soporte"]
        has_priority = case.get("expected_priority") in ["Baja", "Normal", "Alta", "Urgente"]

        is_success = is_valid_project and has_tracker and has_priority
        if is_success:
            passed_cases += 1

        details.append(
            {
                "id": cid,
                "description": desc,
                "expected_project": expected_proj,
                "tracker": case.get("expected_tracker"),
                "priority": case.get("expected_priority"),
                "status": "PASSED" if is_success else "FAILED",
            }
        )

    accuracy = (passed_cases / total_cases) * 100 if total_cases > 0 else 0.0
    return {
        "subsystem": "Enrutamiento a Proyectos y Creación",
        "total": total_cases,
        "passed": passed_cases,
        "accuracy_pct": round(accuracy, 2),
        "details": details,
    }


def run_all_evaluations(save_report: bool = True) -> dict[str, Any]:
    """Ejecuta todas las evaluaciones y consolida el reporte global."""
    print("=" * 80)
    print("🚀 EJECUTANDO EVALUACIÓN DEL AGENTE LANGGRAPH (CASOS REALISTAS)")
    print("=" * 80)

    clarif_results = evaluate_clarification_subsystem()
    dedup_results = evaluate_deduplication_subsystem()
    routing_results = evaluate_routing_and_creation_subsystem()

    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "clarification_accuracy": clarif_results["accuracy_pct"],
            "deduplication_f1": dedup_results["f1_score"],
            "deduplication_precision": dedup_results["precision_pct"],
            "deduplication_recall": dedup_results["recall_pct"],
            "project_routing_accuracy": routing_results["accuracy_pct"],
        },
        "subsystems": {
            "clarification": clarif_results,
            "deduplication": dedup_results,
            "routing_and_creation": routing_results,
        },
    }

    # Imprimir resumen en consola
    print(f"\n1. {clarif_results['subsystem']}:")
    print(f"   - Casos pasados: {clarif_results['passed']}/{clarif_results['total']}")
    print(f"   - Precisión / Eficiencia: {clarif_results['accuracy_pct']}%")

    print(f"\n2. {dedup_results['subsystem']}:")
    print(f"   - Casos pasados: {dedup_results['passed']}/{dedup_results['total']}")
    print(f"   - Precision: {dedup_results['precision_pct']}%")
    print(f"   - Recall: {dedup_results['recall_pct']}%")
    print(f"   - F1-Score: {dedup_results['f1_score']}%")

    print(f"\n3. {routing_results['subsystem']}:")
    print(f"   - Casos pasados: {routing_results['passed']}/{routing_results['total']}")
    print(f"   - Exactitud de Enrutamiento: {routing_results['accuracy_pct']}%")

    if save_report:
        report_path = PROJECT_ROOT / "tests" / "agent_eval" / "evaluation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n📄 Reporte guardado exitosamente en: {report_path}")

        md_path = PROJECT_ROOT / "tests" / "agent_eval" / "evaluation_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Reporte de Evaluación del Agente LangGraph\n\n")
            f.write(f"**Fecha y hora:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Resumen de Métricas\n\n")
            f.write("| Subsistema Evaluado | Métrica Principal | Resultado |\n")
            f.write("| :--- | :--- | :--- |\n")
            f.write(
                f"| Clarificación y Suficiencia | Precisión de Enrutamiento | **{clarif_results['accuracy_pct']}%** |\n"
            )
            f.write(
                f"| Deduplicación y Triaje | F1-Score (Equilibrio Precision/Recall) | **{dedup_results['f1_score']}%** |\n"
            )
            f.write(
                f"| Deduplicación y Triaje | Control de Falsos Positivos (Precision) | **{dedup_results['precision_pct']}%** |\n"
            )
            f.write(
                f"| Enrutamiento a Proyectos | Exactitud de Asignación | **{routing_results['accuracy_pct']}%** |\n\n"
            )

            f.write("## Detalle de Casos por Subsistema\n\n")
            subsystems_dict: dict[str, Any] = report["subsystems"]
            for sub_name, sub_data in subsystems_dict.items():
                f.write(f"### {sub_data['subsystem']}\n")
                f.write("| ID | Descripción | Estado |\n")
                f.write("| :--- | :--- | :--- |\n")
                for d in sub_data["details"]:
                    f.write(f"| `{d['id']}` | {d['description']} | **{d['status']}** |\n")
                f.write("\n")
        print(f"📄 Reporte Markdown guardado en: {md_path}")

    print("\n" + "=" * 80)
    print("✅ EVALUACIÓN COMPLETADA CON ÉXITO")
    print("=" * 80)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runner de evaluación del agente LangGraph.")
    parser.add_argument("--no-save", action="store_true", help="No guardar reportes en disco.")
    args = parser.parse_args()

    run_all_evaluations(save_report=not args.no_save)
