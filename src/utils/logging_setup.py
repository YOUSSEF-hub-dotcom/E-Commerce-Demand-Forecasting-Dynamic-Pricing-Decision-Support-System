"""Centralized logging configuration shared by all pipeline modules.

Every module of the E-Commerce Dynamic Pricing & Demand Forecasting Engine
routes its logs through `get_logger`. A single StreamHandler is registered per
logger to avoid duplicated handlers across re-imports (notebook re-runs).
"""

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger instance with a single stdout handler.

    Parameters
    ----------
    name : str
        Logger name, conventionally the module ``__name__``.
    level : int, optional
        Minimum logging level (default ``logging.INFO``).

    Returns
    -------
    logging.Logger
        Configured logger. Idempotent: a second call returns the same logger
        without stacking additional handlers.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger