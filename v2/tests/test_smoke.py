"""Smoke tests for the staged pipeline.

These tests verify that:
- the dataset acquisition pipeline can produce an `aclImdb` directory, and
- v0/v1 artifacts (if present) can be loaded for inference.
"""

from __future__ import annotations

import os
import joblib
from pathlib import Path

import pytest

from v0.data import ensure_aclImdb


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_data_pipeline_returns_acl_dir():
    """Ensure dataset acquisition returns a valid `aclImdb` directory."""
    acl = ensure_aclImdb()
    assert os.path.isdir(acl)
    assert os.path.isdir(os.path.join(acl, "train", "pos"))
    assert os.path.isdir(os.path.join(acl, "test", "neg"))


def test_v0_artifact_loads_and_predicts():
    """Ensure the latest v0 artifact can be loaded and used for a prediction."""
    paths = sorted((PROJECT_ROOT / "v0").glob("best_model_v0_*.joblib"))
    if not paths:
        pytest.skip("No v0 artifact found. Run v0 first to generate best_model_v0_*.joblib.")
    p = str(paths[-1])

    d = joblib.load(p)
    m = d["model"]
    pred = int(m.predict(["this movie is great"])[0])
    assert pred in (0, 1)


def test_v1_artifact_loads():
    """Ensure the latest v1 artifact can be loaded successfully."""
    paths = sorted((PROJECT_ROOT / "v1").glob("best_model_v1_nn_*.joblib"))
    if not paths:
        pytest.skip("No v1 artifact found. Run v1 first to generate best_model_v1_nn_*.joblib.")
    p = str(paths[-1])

    d = joblib.load(p)
    assert "vectorizer" in d
    assert "svd" in d
    assert "nn" in d
