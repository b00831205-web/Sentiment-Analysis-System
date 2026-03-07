"""Artifact loading utilities for v2.

v2 does not retrain models. Instead, it loads the best artifacts produced by v0
and v1 and exposes them to the CLI and server prediction endpoints.
"""


from __future__ import annotations

import glob
import os
import joblib
import logging

log = logging.getLogger("v2.model_loader")

def _latest(pattern: str) -> str:
    """Return the most recent file matching a glob pattern.

    Args:
        pattern: Glob pattern (e.g., "v0/best_model_v0_*.joblib").

    Returns:
        Path of the last match in sorted order.

    Raises:
        FileNotFoundError: If no files match the pattern.
    """
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files match: {pattern}")
    return matches[-1]

def load_v0_model(project_root: str) -> dict:
    """Load the latest v0 artifact.

    Args:
        project_root: Project root directory containing `v0/`.

    Returns:
        A dict with the following keys:
            - "path": Path to the artifact file.
            - "bundle": The full loaded artifact dictionary.
            - "model": The trained sklearn pipeline.
            - "best_name": Name of the best-performing model.
            
    Raises:
        FileNotFoundError: If the expected artifact file is not found.
        KeyError: If the loaded artifact does not contain required keys.
    """
    # expects v0/best_model_v0_*.joblib
    p = _latest(os.path.join(project_root, "v0", "best_model_v0_*.joblib"))
    log.info(f"Loading v0 model: {p}")
    d = joblib.load(p)
    return {"path": p, "bundle": d, "model": d["model"], "best_name": d.get("best_name", "v0")}

def load_v1_model(project_root: str) -> dict:
    """Load the latest v1 artifact.

    Args:
        project_root: Project root directory containing `v1/`.

    Returns:
        A dict with the following keys:
            - "path": Path to the artifact file.
            - "bundle": The full loaded artifact dictionary (vectorizer, SVD, NN weights, etc.).

    Raises:
        FileNotFoundError: If the expected artifact file is not found.
    """
    # expects v1/best_model_v1_nn_*.joblib
    p = _latest(os.path.join(project_root, "v1", "best_model_v1_nn_*.joblib"))
    log.info(f"Loading v1 model: {p}")
    d = joblib.load(p)
    return {"path": p, "bundle": d}
