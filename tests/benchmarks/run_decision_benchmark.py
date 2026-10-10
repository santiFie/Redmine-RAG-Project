"""
tests/benchmarks/run_decision_benchmark.py
==========================================
Runner automatizado para el benchmark multimodelo de Decision Models (Issue #12).

Ejecuta en paralelo controlado los casos de prueba curados contra:
- OpenRouter: microsoft/microsoft-decision-1, typesafe/jev-1.13, openai/gpt-6-luna-decisions
- Local: Kev-0.8B (http://127.0.0.1:8009/v1/systemone)
- GCP VM: Kev-4B (http://<GCP_HOST>:8009/v1/systemone)

Maneja de manera resiliente modelos no configurados o endpoints fuera de línea
sin interrumpir la evaluación de los demás.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Asegurar que el directorio raíz del proyecto esté en el sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tests.benchmarks.config import (
    BenchmarkModelConfig,
    INTENT_QUESTIONS,
    SAFE_QUERY_QUESTIONS,
    get_benchmark_models,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("decision_benchmark")


def load_dataset(dataset_path: Path) -> dict[str, Any]:
    """Carga el dataset curado de Ground Truth."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset en: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_percentile(values: list[float], percentile: float) -> float:
    """Calcula el percentil indicado a partir de una lista de floats."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (percentile / 100.0)
    f = int(k)
    c = f + 1
    if c < len(sorted_vals):
        return round(sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f]), 2)
    return round(sorted_vals[f], 2)


async def evaluate_safe_query(
    case: dict[str, Any],
    model_key: str,
    model_cfg: BenchmarkModelConfig,
    semaphore: asyncio.Semaphore,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Evalúa un caso de safe_query contra un modelo específico."""
    user_input = case["user_input"]
    expected = case["expected_is_safe"]

    if not model_cfg.is_configured():
        return {
            "model_key": model_key,
            "display_name": model_cfg.display_name,
            "status": "unconfigured",
            "prediction": None,
            "is_correct": False,
            "noul_score": None,
            "confidence": None,
            "latency_ms": 0.0,
            "error": "Endpoint o credencial no configurada en variables de entorno.",
        }

    if dry_run:
        # Simulación controlada para verificación de tubería
        await asyncio.sleep(0.01)
        return {
            "model_key": model_key,
            "display_name": model_cfg.display_name,
            "status": "success",
            "prediction": expected,
            "is_correct": True,
            "noul_score": 0.95 if expected else 0.05,
            "confidence": 0.9,
            "latency_ms": 12.5,
            "error": None,
        }

    async with semaphore:
        start_time = time.perf_counter()
        try:
            classifier = model_cfg.create_classifier()
            response = await classifier.ainvoke(
                {
                    "state": user_input,
                    "questions": SAFE_QUERY_QUESTIONS,
                }
            )
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

            noul_ans = response.nouls.get("safe_to_run")
            score = noul_ans.noul if noul_ans is not None else None
            is_safe = (score >= 0.5) if score is not None else False
            confidence = getattr(noul_ans, "confidence", None)

            return {
                "model_key": model_key,
                "display_name": model_cfg.display_name,
                "status": "success",
                "prediction": is_safe,
                "is_correct": (is_safe == expected),
                "noul_score": score,
                "confidence": confidence,
                "latency_ms": elapsed_ms,
                "error": None,
            }
        except Exception as e:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.warning(
                f"Error evaluando safe_query con {model_key} para caso {case['id']}: {e}"
            )
            return {
                "model_key": model_key,
                "display_name": model_cfg.display_name,
                "status": "error",
                "prediction": None,
                "is_correct": False,
                "noul_score": None,
                "confidence": None,
                "latency_ms": elapsed_ms,
                "error": str(e),
            }


async def evaluate_intent(
    case: dict[str, Any],
    model_key: str,
    model_cfg: BenchmarkModelConfig,
    semaphore: asyncio.Semaphore,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Evalúa un caso de intent classification contra un modelo específico."""
    user_input = case["user_input"]
    expected = case["expected_intent"]
    acceptable_sec = case.get("acceptable_secondary")

    if not model_cfg.is_configured():
        return {
            "model_key": model_key,
            "display_name": model_cfg.display_name,
            "status": "unconfigured",
            "prediction": None,
            "is_correct": False,
            "is_acceptable": False,
            "confidence": None,
            "probabilities": {},
            "latency_ms": 0.0,
            "error": "Endpoint o credencial no configurada en variables de entorno.",
        }

    if dry_run:
        await asyncio.sleep(0.01)
        return {
            "model_key": model_key,
            "display_name": model_cfg.display_name,
            "status": "success",
            "prediction": expected,
            "is_correct": True,
            "is_acceptable": True,
            "confidence": 0.88,
            "probabilities": {expected: 0.88, "general": 0.1},
            "latency_ms": 15.0,
            "error": None,
        }

    async with semaphore:
        start_time = time.perf_counter()
        try:
            classifier = model_cfg.create_classifier()
            response = await classifier.ainvoke(
                {
                    "state": user_input,
                    "questions": INTENT_QUESTIONS,
                }
            )
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

            choice_ans = response.choices.get("intent")
            choice = choice_ans.choice if choice_ans is not None else None
            confidence = getattr(choice_ans, "confidence", None)
            probabilities = getattr(choice_ans, "probabilities", {}) or {}

            is_exact = choice == expected
            is_acceptable = is_exact or (acceptable_sec is not None and choice == acceptable_sec)

            return {
                "model_key": model_key,
                "display_name": model_cfg.display_name,
                "status": "success",
                "prediction": choice,
                "is_correct": is_exact,
                "is_acceptable": is_acceptable,
                "confidence": confidence,
                "probabilities": probabilities,
                "latency_ms": elapsed_ms,
                "error": None,
            }
        except Exception as e:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.warning(f"Error evaluando intent con {model_key} para caso {case['id']}: {e}")
            return {
                "model_key": model_key,
                "display_name": model_cfg.display_name,
                "status": "error",
                "prediction": None,
                "is_correct": False,
                "is_acceptable": False,
                "confidence": None,
                "probabilities": {},
                "latency_ms": elapsed_ms,
                "error": str(e),
            }


async def run_benchmark(
    dataset_path: Path,
    output_path: Path,
    selected_model_keys: list[str] | None = None,
    concurrency: int = 5,
    smoke: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Orquesta la ejecución completa del benchmark en paralelo."""
    dataset = load_dataset(dataset_path)
    all_models = get_benchmark_models()

    if selected_model_keys:
        models = {k: v for k, v in all_models.items() if k in selected_model_keys}
    else:
        models = all_models

    safe_cases = dataset["safe_query_cases"]
    intent_cases = dataset["intent_cases"]

    if smoke:
        logger.info(
            "🧪 Modo SMOKE activado: seleccionando subconjunto mínimo de casos (4 en total)"
        )
        safe_cases = [c for c in safe_cases if c["difficulty"] == "clear"][:1] + [
            c for c in safe_cases if c["difficulty"] == "boundary"
        ][:1]
        intent_cases = [c for c in intent_cases if c["difficulty"] == "clear"][:1] + [
            c for c in intent_cases if c["difficulty"] == "boundary"
        ][:1]

    semaphore = asyncio.Semaphore(concurrency)
    logger.info(
        f"Iniciando benchmark: {len(safe_cases)} safe_query y {len(intent_cases)} intent "
        f"contra {len(models)} modelos con concurrencia={concurrency}"
    )

    # 1. Evaluar Safe Query en paralelo
    safe_tasks = []
    for case in safe_cases:
        for m_key, m_cfg in models.items():
            safe_tasks.append(
                (case, m_key, evaluate_safe_query(case, m_key, m_cfg, semaphore, dry_run))
            )

    safe_eval_outputs = await asyncio.gather(*[t[2] for t in safe_tasks])

    # Agrupar resultados por caso
    safe_cases_map: dict[str, dict[str, Any]] = {}
    for (case, m_key, _), eval_out in zip(safe_tasks, safe_eval_outputs):
        case_id = case["id"]
        if case_id not in safe_cases_map:
            safe_cases_map[case_id] = {
                "case_info": case,
                "model_results": {},
            }
        safe_cases_map[case_id]["model_results"][m_key] = eval_out

    # 2. Evaluar Intent en paralelo
    intent_tasks = []
    for case in intent_cases:
        for m_key, m_cfg in models.items():
            intent_tasks.append(
                (case, m_key, evaluate_intent(case, m_key, m_cfg, semaphore, dry_run))
            )

    intent_eval_outputs = await asyncio.gather(*[t[2] for t in intent_tasks])

    intent_cases_map: dict[str, dict[str, Any]] = {}
    for (case, m_key, _), eval_out in zip(intent_tasks, intent_eval_outputs):
        case_id = case["id"]
        if case_id not in intent_cases_map:
            intent_cases_map[case_id] = {
                "case_info": case,
                "model_results": {},
            }
        intent_cases_map[case_id]["model_results"][m_key] = eval_out

    # 3. Consolidar Estadísticas por Modelo
    model_stats: dict[str, Any] = {}
    for m_key, m_cfg in models.items():
        # Safe query stats
        m_safe = [
            c_data["model_results"][m_key]
            for c_data in safe_cases_map.values()
            if m_key in c_data["model_results"]
        ]
        safe_success = [r for r in m_safe if r["status"] == "success"]
        safe_correct = [r for r in safe_success if r["is_correct"]]
        safe_clear = [
            r
            for c_data in safe_cases_map.values()
            if (r := c_data["model_results"].get(m_key))
            and c_data["case_info"]["difficulty"] == "clear"
            and r["status"] == "success"
        ]
        safe_boundary = [
            r
            for c_data in safe_cases_map.values()
            if (r := c_data["model_results"].get(m_key))
            and c_data["case_info"]["difficulty"] == "boundary"
            and r["status"] == "success"
        ]

        # Intent stats
        m_intent = [
            c_data["model_results"][m_key]
            for c_data in intent_cases_map.values()
            if m_key in c_data["model_results"]
        ]
        intent_success = [r for r in m_intent if r["status"] == "success"]
        intent_correct = [r for r in intent_success if r["is_correct"]]
        intent_acceptable = [r for r in intent_success if r.get("is_acceptable", False)]
        intent_clear = [
            r
            for c_data in intent_cases_map.values()
            if (r := c_data["model_results"].get(m_key))
            and c_data["case_info"]["difficulty"] == "clear"
            and r["status"] == "success"
        ]
        intent_boundary = [
            r
            for c_data in intent_cases_map.values()
            if (r := c_data["model_results"].get(m_key))
            and c_data["case_info"]["difficulty"] == "boundary"
            and r["status"] == "success"
        ]

        # Latencies
        all_latencies = [
            r["latency_ms"] for r in (safe_success + intent_success) if r["latency_ms"] > 0
        ]
        mean_latency = round(sum(all_latencies) / len(all_latencies), 2) if all_latencies else 0.0
        p50_latency = calculate_percentile(all_latencies, 50.0)
        p95_latency = calculate_percentile(all_latencies, 95.0)

        # Status
        total_attempts = len(m_safe) + len(m_intent)
        total_errors = len(
            [r for r in (m_safe + m_intent) if r["status"] in ("error", "unconfigured")]
        )
        online_status = (
            "offline" if (total_errors == total_attempts and total_attempts > 0) else "online"
        )

        model_stats[m_key] = {
            "display_name": m_cfg.display_name,
            "provider": m_cfg.provider,
            "model_id": m_cfg.model_id,
            "online_status": online_status,
            "total_calls": total_attempts,
            "successful_calls": len(safe_success) + len(intent_success),
            "error_calls": total_errors,
            "safe_query": {
                "total": len(safe_cases),
                "accuracy_overall": round(len(safe_correct) / len(safe_success), 4)
                if safe_success
                else 0.0,
                "accuracy_clear": round(
                    len([r for r in safe_clear if r["is_correct"]]) / len(safe_clear), 4
                )
                if safe_clear
                else 0.0,
                "accuracy_boundary": round(
                    len([r for r in safe_boundary if r["is_correct"]]) / len(safe_boundary), 4
                )
                if safe_boundary
                else 0.0,
            },
            "intent": {
                "total": len(intent_cases),
                "accuracy_overall": round(len(intent_correct) / len(intent_success), 4)
                if intent_success
                else 0.0,
                "accuracy_acceptable": round(len(intent_acceptable) / len(intent_success), 4)
                if intent_success
                else 0.0,
                "accuracy_clear": round(
                    len([r for r in intent_clear if r["is_correct"]]) / len(intent_clear), 4
                )
                if intent_clear
                else 0.0,
                "accuracy_boundary": round(
                    len([r for r in intent_boundary if r["is_correct"]]) / len(intent_boundary), 4
                )
                if intent_boundary
                else 0.0,
            },
            "latency_ms": {
                "mean": mean_latency,
                "p50": p50_latency,
                "p95": p95_latency,
            },
        }

    # Estructura de salida final
    benchmark_output = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "smoke_mode": smoke,
            "dry_run": dry_run,
            "concurrency": concurrency,
            "total_safe_cases": len(safe_cases),
            "total_intent_cases": len(intent_cases),
            "models_evaluated": list(models.keys()),
        },
        "models_summary": model_stats,
        "safe_query_results": list(safe_cases_map.values()),
        "intent_results": list(intent_cases_map.values()),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_output, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Benchmark finalizado exitosamente. Resultados guardados en: {output_path}")
    return benchmark_output


def main():
    parser = argparse.ArgumentParser(
        description="Runner de Benchmark para Decision Models (Issue #12)"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_PROJECT_ROOT / "tests" / "benchmarks" / "datasets" / "ground_truth.json",
        help="Ruta al dataset de Ground Truth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_PROJECT_ROOT
        / "tests"
        / "benchmarks"
        / "results"
        / "decision_benchmark_results.json",
        help="Ruta del archivo de salida JSON",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Lista de claves de modelos separadas por coma a evaluar (ej: ms-decision-1,jev-1.13)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Límite de peticiones asíncronas concurrentes",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Ejecuta una corrida mínima (smoke test) de 4 casos para verificar funcionamiento",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simula inferencias sin consultar APIs externas",
    )

    args = parser.parse_args()
    selected_models = [m.strip() for m in args.models.split(",")] if args.models else None

    asyncio.run(
        run_benchmark(
            dataset_path=args.dataset,
            output_path=args.output,
            selected_model_keys=selected_models,
            concurrency=args.concurrency,
            smoke=args.smoke,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
