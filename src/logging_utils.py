"""Logger helper: prefer loguru, fall back to stdlib logging."""

from __future__ import annotations


def get_logger(name: str = "credit_scoring"):
    try:
        from loguru import logger

        return logger
    except Exception:  # pragma: no cover
        import logging

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
        )
        return logging.getLogger(name)
