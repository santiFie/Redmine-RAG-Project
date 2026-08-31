"""
tests/ragas/conftest.py
=======================
Configuración de pytest para la suite de evaluación Ragas.

Registra la opción ``--suites`` que permite seleccionar qué suites
ejecutar sin correr toda la evaluación.

Nota: pytest_addoption DEBE residir en conftest.py — no en módulos de test.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Suites válidas (sincronizar con _SUITES_VALIDAS en test_rag_ragas.py)
# ---------------------------------------------------------------------------

_SUITES_VALIDAS = frozenset({"base", "automerging", "hybrid", "metadata_filter"})


def pytest_addoption(parser: pytest.Parser) -> None:
    """
    Registra la opción ``--suites`` en pytest.

    Acepta una lista de suites separadas por coma:
      base, automerging, hybrid, metadata_filter

    Ejemplos:
      --suites hybrid
      --suites hybrid,automerging

    Si se omite el flag, se consulta la variable de entorno ``RAGAS_SUITES``.
    Sin ninguna de las dos fuentes, se ejecutan todas las suites.
    """
    parser.addoption(
        "--suites",
        action="store",
        default=None,
        help=(
            "Suites Ragas a ejecutar, separadas por coma: "
            "base, automerging, hybrid, metadata_filter. "
            "Sin valor ejecuta todas. Fallback: variable de entorno RAGAS_SUITES."
        ),
    )
