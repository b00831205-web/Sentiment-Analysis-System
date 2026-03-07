"""Configuration loader utilities.

This module provides a small helper to load JSON configuration from the project
root (e.g., `config.json`). It is designed to be optional and robust: if the file
does not exist or is invalid, the loader returns an empty dict and the program
falls back to CLI defaults.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_config(project_root: str | Path, config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load configuration from a JSON file.

    Priority:
        - If `config_path` is provided, load from that path.
        - Otherwise, load from `<project_root>/config.json`.

    Args:
        project_root: Project root directory.
        config_path: Optional explicit config file path.

    Returns:
        A dict parsed from JSON, or {} if the config file is missing/invalid.
    """
    root = Path(project_root)
    path = Path(config_path) if config_path is not None else (root / "config.json")

    try:
        if not path.exists():
            return {}
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        # Be robust: config is optional; fall back to defaults.
        return {}
