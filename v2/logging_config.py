"""Logging configuration for v2.

This module centralizes logging setup (console + file) to keep CLI/server output
consistent and easy to debug during evaluation.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

def setup_logging(log_dir: str | None = None, level: str = "INFO") -> None:
    """Configure logging handlers and formatting for v2.

    Args:
        log_dir: Directory for log files. If None, defaults to `<project_root>/logs`.
        level: Logging level name (e.g., "INFO", "DEBUG").

    Returns:
        None.

    Side Effects:
        Creates the log directory if needed and writes to `v2.log`.
        
    Notes:
        If logging is already configured elsewhere, `basicConfig` may have no effect.
    """

    lvl = getattr(logging, level.upper(), logging.INFO)

    if log_dir is None:
        # project_root/logs
        project_root = Path(__file__).resolve().parents[1]
        log_dir = str(project_root / "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_path = Path(log_dir) / "v2.log"

    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_path, encoding="utf-8"),
    ]
    logging.basicConfig(level=lvl, format=fmt, handlers=handlers)
