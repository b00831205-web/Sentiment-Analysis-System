"""Prediction helpers for v2.

This module provides a uniform prediction interface over v0 and v1 artifacts.
It is used by both the CLI and the Flask server.
"""

from __future__ import annotations

import numpy as np
import logging
from pathlib import Path  
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))  


from common.text_utils import clean_text

log = logging.getLogger("v2.predict")

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for NumPy arrays.

    Args:
        x: Input array.

    Returns:
        Sigmoid-transformed array with the same shape.
    """
    x = np.clip(x, -30, 30)
    return 1.0 / (1.0 + np.exp(-x))

def predict_v0(model, text: str) -> dict:
    """Run inference using a v0 sklearn pipeline.

    Args:
        model: Trained sklearn pipeline.
        text: Raw input text.

    Returns:
        Dict with keys: `label` (0/1) and `prob_pos` (float or None).
    """
    # sklearn pipeline: predict + predict_proba (if exists)
    pred = int(model.predict([text])[0])
    prob = None
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba([text])[0, 1])
    return {"label": pred, "prob_pos": prob}

def predict_v1(bundle: dict, text: str) -> dict:
    """Run inference using a v1 neural network bundle.

    Args:
        bundle: Dict containing the fitted preprocessing objects and NN parameters
            (e.g., "vectorizer", "svd", "mu", "sigma", and "nn" weights/biases).
        text: Raw input text.

    Returns:
        Dict with keys:
            - "label": Predicted class (0/1)
            - "prob_pos": Positive-class probability in [0, 1]

    Raises:
        KeyError: If required keys are missing from the loaded bundle.
    """

    # bundle structure saved by your v1_auto.py
    vec = bundle["vectorizer"]
    svd = bundle["svd"]
    mu = bundle["mu"]
    sigma = bundle["sigma"]
    W1 = bundle["nn"]["W1"]
    b1 = bundle["nn"]["b1"]
    W2 = bundle["nn"]["W2"]
    b2 = bundle["nn"]["b2"]

    # v1 vectorizer uses preprocessor=common.text_utils.clean_text already,
    # but calling clean_text here is safe and consistent.
    _ = clean_text(text)

    X = vec.transform([text])
    Z = svd.transform(X).astype("float32")
    Z = (Z - mu) / sigma
    A = np.tanh(Z @ W1 + b1)
    y = _sigmoid(A @ W2 + b2)[0, 0]
    pred = int(y >= 0.5)
    return {"label": pred, "prob_pos": float(y)}

def predict(model_kind: str, v0_ctx: dict | None, v1_ctx: dict | None, text: str) -> dict:
    """Dispatch prediction to the requested model kind.

    Args:
        model_kind: "v0" or "v1".
        v0_ctx: Loaded v0 context (from `load_v0_model`), or None.
        v1_ctx: Loaded v1 context (from `load_v1_model`), or None.
        text: Raw input text.

    Returns:
        Prediction dict augmented with model metadata.

    Raises:
        ValueError: If `model_kind` is invalid or the required artifact is not loaded.
    """
    model_kind = model_kind.lower().strip()
    if model_kind == "v0":
        if not v0_ctx:
            raise ValueError("v0 model not loaded")
        out = predict_v0(v0_ctx["model"], text)
        out["model"] = "v0"
        out["model_path"] = v0_ctx["path"]
        out["model_name"] = v0_ctx.get("best_name")
        return out

    if model_kind == "v1":
        if not v1_ctx:
            raise ValueError("v1 model not loaded")
        out = predict_v1(v1_ctx["bundle"], text)
        out["model"] = "v1"
        out["model_path"] = v1_ctx["path"]
        out["model_name"] = v1_ctx["bundle"].get("model_type", "v1")
        return out

    raise ValueError("model must be one of: v0, v1")
