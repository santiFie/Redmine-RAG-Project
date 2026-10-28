#!/usr/bin/env python3
"""
tests/agent_eval/audit_llm_judge.py
===================================
Script de Auditoría Interactiva con LLM-as-a-Judge (Capa 3).

Permite auditar de forma selectiva únicamente aquellos ejemplos donde la evaluación
previa de Decision Models (JEV / OpenRouter) haya arrojado un score menor a 0.60
(o fallos deterministas por salida vacía).

Regla Estricta:
NUNCA se ejecuta de forma automática. Siempre solicita confirmación explícita al
usuario por terminal antes de consumir tokens generativos de LLM.
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

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langsmith import Client
from src.evaluation.llm_judge_evaluators import (
    LLM_JUDGE_MODEL,
    LLM_JUDGE_PROVIDER,
    ClarificationJudgeAudit,
    QAExtractionJudgeAudit,
    TriageJudgeAudit,
    audit_clarification_with_llm,
    audit_qa_extraction_with_llm,
    audit_triage_with_llm,
    sync_judge_prompt_to_hub,
)


def confirm_execution(prompt: str) -> bool:
    """Solicita autorización explícita al usuario por terminal."""
    try:
        response = input(f"\n{prompt} [s/N]: ").strip().lower()
        return response in ("s", "si", "y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


async def audit_flagged_cases(
    flagged_cases: list[dict[str, Any]],
    client: Client | None = None,
    provider: str = LLM_JUDGE_PROVIDER,
    model: str = LLM_JUDGE_MODEL,
) -> list[dict[str, Any]]:
    """Ejecuta la auditoría LLM-as-a-Judge sobre los casos autorizados."""
    print(f"\n🤖 Iniciando auditoría con LLM-as-a-Judge ({provider} / {model})...")
    results = []

    for idx, case in enumerate(flagged_cases, 1):
        case_id = case.get("id", f"case-{idx}")
        user_input = case.get("inputs", {})
        question_text = case.get("question_text", "")
        expected = case.get("expected", {})
        jev_score = case.get("decision_model_score", 0.0)
        case_type = case.get("type", case.get("suite", "clarification"))

        print(
            f"\n[{idx}/{len(flagged_cases)}] Auditando caso: {case_id} ({case_type.upper()} - Score JEV previo: {jev_score})..."
        )

        if "qa" in case_type.lower():
            qa_audit: QAExtractionJudgeAudit = await audit_qa_extraction_with_llm(
                inputs=user_input,
                outputs=case.get("outputs", {}),
                expected=expected,
                decision_model_score=jev_score,
                provider=provider,
                model=model,
            )
            overall_score = qa_audit.average_score
            audit_dump = qa_audit.to_audit_dict()

            print(f"  📊 Veredicto LLM Judge QA: {overall_score}")
            print(
                f"     • Completitud ({qa_audit.completeness_score}): {qa_audit.completeness_analysis}"
            )
            print(
                f"     • Cero Alucinación ({qa_audit.no_hallucination_score}): {qa_audit.hallucination_analysis}"
            )
            print(
                f"     • Suficiencia ({qa_audit.sufficiency_score}): {qa_audit.sufficiency_analysis}"
            )
            print(
                f"     • Tipo Incidente ({qa_audit.incident_type_score}): {qa_audit.incident_type_analysis}"
            )
            if qa_audit.divergence_rationale:
                print(f"  ⚠️ Causa Raíz de Divergencia: {qa_audit.divergence_rationale}")
            print(f"  💡 Recomendación Accionable: {qa_audit.actionable_feedback}")

        elif "triage" in case_type.lower():
            triage_audit: TriageJudgeAudit = await audit_triage_with_llm(
                inputs=user_input,
                outputs=case.get("outputs", {}),
                expected=expected,
                decision_model_score=jev_score,
                provider=provider,
                model=model,
            )
            overall_score = triage_audit.average_score
            audit_dump = triage_audit.to_audit_dict()

            print(f"  📊 Veredicto LLM Judge Triaje: {overall_score}")
            print(
                f"     • Fidelidad RAG ({triage_audit.rag_fidelity_score}): {triage_audit.rag_fidelity_analysis}"
            )
            print(
                f"     • Reglas Deduplicación ({triage_audit.deduplication_rules_score}): {triage_audit.deduplication_rules_analysis}"
            )
            print(
                f"     • Enrutamiento Proyecto ({triage_audit.project_routing_score}): {triage_audit.project_routing_analysis}"
            )
            if triage_audit.divergence_rationale:
                print(f"  ⚠️ Causa Raíz de Divergencia: {triage_audit.divergence_rationale}")
            print(f"  💡 Recomendación Accionable: {triage_audit.actionable_feedback}")

        else:
            clarif_audit: ClarificationJudgeAudit = await audit_clarification_with_llm(
                inputs=user_input,
                question_text=question_text,
                expected=expected,
                decision_model_score=jev_score,
                provider=provider,
                model=model,
                client=client,
            )
            overall_score = clarif_audit.average_score
            audit_dump = clarif_audit.to_audit_dict()

            print(
                f"  📊 Veredicto LLM Judge Clarificación: Promedio={overall_score} (Mínimo={clarif_audit.min_dimension_score})"
            )
            print(
                f"     1. Cobertura ({clarif_audit.coverage_score}): {clarif_audit.coverage_analysis}"
            )
            print(
                f"     2. Alcance ({clarif_audit.no_out_of_scope_score}): {clarif_audit.no_out_of_scope_analysis}"
            )
            print(
                f"     3. Cero Redundancia ({clarif_audit.no_redundancy_score}): {clarif_audit.no_redundancy_analysis}"
            )
            print(
                f"     4. Reglas Dominio ({clarif_audit.rule_compliance_score}): {clarif_audit.rule_compliance_analysis}"
            )
            print(
                f"     5. Tono/Claridad ({clarif_audit.tone_clarity_score}): {clarif_audit.tone_clarity_analysis}"
            )
            if clarif_audit.divergence_from_decision_model_rationale:
                print(
                    f"  ⚠️ Causa Raíz de Divergencia: {clarif_audit.divergence_from_decision_model_rationale}"
                )
            print(f"  💡 Recomendación Accionable: {clarif_audit.actionable_feedback}")

        # Registrar en LangSmith si hay run_id disponible
        run_id = case.get("run_id")
        if client and run_id:
            try:
                client.create_feedback(
                    run_id=run_id,
                    key="llm_judge_audit_score",
                    score=overall_score,
                    comment=f"Auditoría profunda ({case_type}): Score={overall_score}. Feedback: {audit_dump.get('actionable_feedback', '')}",
                    extra=audit_dump,
                )
                print(f"  ✅ Feedback registrado en LangSmith para el run {run_id}.")
            except Exception as e:
                print(f"  ⚠️ No se pudo registrar feedback en LangSmith: {e}")

        results.append(
            {
                "case_id": case_id,
                "decision_model_score": jev_score,
                "llm_judge_audit": audit_dump,
            }
        )

    return results


async def run_interactive_audit_from_dataset(
    dataset_file: Path,
    threshold: float = 0.60,
    provider: str = LLM_JUDGE_PROVIDER,
    model: str = LLM_JUDGE_MODEL,
):
    """Audita casos de prueba leyendo directamente un archivo de dataset local."""
    if not dataset_file.exists():
        print(f"❌ No se encontró el dataset en: {dataset_file}")
        return

    with open(dataset_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = data.get("examples", [])
    print(f"📂 Dataset cargado: {dataset_file.name} ({len(examples)} ejemplos).")

    # Para fines de demostración/auditoría manual
    print("\nParámetros configurados:")
    print(f"  - Umbral de activación: Score JEV < {threshold}")
    print(f"  - Modelo Juez: {provider} / {model}")


def main():
    parser = argparse.ArgumentParser(
        description="Auditoría Interactiva con LLM-as-a-Judge para Decision Models (Capa 3)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ask_clarification_ground_truth.json",
        help="Archivo JSON de dataset en tests/agent_eval/datasets/",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.60,
        help="Umbral de score de Decision Models para filtrar casos anómalos (default: 0.60).",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=LLM_JUDGE_PROVIDER,
        help=f"Proveedor del LLM Juez (default: {LLM_JUDGE_PROVIDER}).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=LLM_JUDGE_MODEL,
        help=f"Modelo del LLM Juez (default: {LLM_JUDGE_MODEL}).",
    )

    parser.add_argument(
        "--sync-prompt",
        action="store_true",
        help="Publica o actualiza el prompt de la rúbrica del juez en el Prompt Hub de LangSmith.",
    )

    args = parser.parse_args()

    if args.sync_prompt:
        client = Client()
        print("📤 Sincronizando prompt de rúbrica hacia LangSmith Prompt Hub...")
        try:
            url = sync_judge_prompt_to_hub(client)
            print(f"  ✅ Prompt publicado exitosamente en LangSmith Hub: {url}")
        except Exception as e:
            print(f"  ❌ Error al sincronizar prompt con LangSmith Hub: {e}")
        return

    dataset_path = PROJECT_ROOT / "tests" / "agent_eval" / "datasets" / args.dataset

    asyncio.run(
        run_interactive_audit_from_dataset(
            dataset_file=dataset_path,
            threshold=args.threshold,
            provider=args.provider,
            model=args.model,
        )
    )


if __name__ == "__main__":
    main()
