"""
src/utils/logging.py
====================
Configuración de logging estructurado con structlog.

Provee un logger centralizado con salida en JSON (producción)
o consola enriquecida con rich (desarrollo).

Uso:
    from src.utils.logging import get_logger

    logger = get_logger(__name__)
    logger.info("issue_fetched", issue_id=42, project="my-project")
"""

from __future__ import annotations

import logging
import os

# TODO: configurar structlog con procesadores adecuados
# import structlog

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def get_logger(name: str):
    """
    Retorna un logger configurado para el módulo dado.

    En producción: salida JSON estructurada.
    En desarrollo: salida coloreada con rich.

    TODO:
        - Configurar structlog.configure() con:
            * timestamper
            * add_log_level
            * JSONRenderer (prod) o ConsoleRenderer (dev)
        - Retornar structlog.get_logger(name)
    """
    # Placeholder: usa el logger estándar de Python hasta que structlog esté configurado
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    return logger
