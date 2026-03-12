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
    
    # Get the root logger and clear existing handlers
    root_logger = logging.getLogger()
    
    # Close and remove all existing handlers
    for handler in root_logger.handlers[:]:
        try:
            handler.close()
        except Exception:
            pass
        root_logger.removeHandler(handler)
    
    # Set root logger level
    root_logger.setLevel(lvl)
    
    # Add handlers
    console_handler = logging.StreamHandler()
    console_handler.setLevel(lvl)
    
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(lvl)
    
    # Set formatter
    formatter = logging.Formatter(fmt)
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
