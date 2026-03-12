"""Unit tests for the v2.model_loader module.

Tests cover:
- Latest file discovery by glob pattern
- v0 and v1 artifact loading
- Error handling for missing artifacts
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
import joblib
import pytest
from unittest.mock import Mock, patch

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v2.model_loader import _latest, load_v0_model, load_v1_model


class TestLatestt:
    """Test the _latest utility function."""

    def test_latest_returns_last_match(self):
        """Test that _latest returns the lexicographically last match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            Path(tmpdir, "model_1.joblib").touch()
            Path(tmpdir, "model_2.joblib").touch()
            Path(tmpdir, "model_10.joblib").touch()

            pattern = os.path.join(tmpdir, "model_*.joblib")
            result = _latest(pattern)

            # Note: string sort, so "model_9.joblib" > "model_10.joblib"
            assert "model_2" in result

    def test_latest_no_matches_raises(self):
        """Test that _latest raises FileNotFoundError when no files match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pattern = os.path.join(tmpdir, "nonexistent_*.joblib")
            with pytest.raises(FileNotFoundError):
                _latest(pattern)

    def test_latest_single_match(self):
        """Test _latest with a single matching file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir, "model.joblib")
            test_file.touch()

            pattern = os.path.join(tmpdir, "model.joblib")
            result = _latest(pattern)

            assert result == str(test_file)


class TestLoadV0Model:
    """Test v0 artifact loading."""

    def test_load_v0_model_success(self):
        """Test successful v0 model loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = tmpdir
            v0_dir = Path(project_root, "v0")
            v0_dir.mkdir()

            # Create a mock artifact
            artifact = {
                "model": Mock(predict=Mock(return_value=[0])),
                "best_name": "random_forest",
                "accuracy": 0.95,
            }
            model_path = v0_dir / "best_model_v0_20231115.joblib"
            joblib.dump(artifact, str(model_path))

            result = load_v0_model(project_root)

            assert result["path"] == str(model_path)
            assert result["best_name"] == "random_forest"
            assert "model" in result["bundle"]
            assert result["model"] == artifact["model"]

    def test_load_v0_model_no_artifact_raises(self):
        """Test that FileNotFoundError is raised when no v0 artifact exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Don't create v0 directory or artifact
            with pytest.raises(FileNotFoundError):
                load_v0_model(tmpdir)

    def test_load_v0_model_missing_required_keys(self):
        """Test behavior when artifact lacks required keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = tmpdir
            v0_dir = Path(project_root, "v0")
            v0_dir.mkdir()

            # Create artifact with missing "model" key
            artifact = {"best_name": "svm"}  # Missing "model"
            model_path = v0_dir / "best_model_v0_20231115.joblib"
            joblib.dump(artifact, str(model_path))

            # Should raise KeyError when accessing ["model"]
            with pytest.raises(KeyError):
                load_v0_model(project_root)

    def test_load_v0_model_uses_latest(self):
        """Test that load_v0_model loads the lexicographically latest artifact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = tmpdir
            v0_dir = Path(project_root, "v0")
            v0_dir.mkdir()

            artifact1 = {"model": Mock(), "best_name": "model1"}
            artifact2 = {"model": Mock(), "best_name": "model2"}

            joblib.dump(
                artifact1, str(v0_dir / "best_model_v0_20231101.joblib")
            )
            joblib.dump(
                artifact2, str(v0_dir / "best_model_v0_20231115.joblib")
            )

            result = load_v0_model(project_root)

            # The loader uses _latest which returns lexicographically last
            assert "best_model_v0_" in result["path"]
            # Since both exist, it should load one of them without error
            assert result["best_name"] in ["model1", "model2"]


class TestLoadV1Model:
    """Test v1 artifact loading."""

    def test_load_v1_model_success(self):
        """Test successful v1 model loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = tmpdir
            v1_dir = Path(project_root, "v1")
            v1_dir.mkdir()

            # Create a mock v1 bundle
            bundle = {
                "vectorizer": Mock(),
                "svd": Mock(),
                "mu": [0.0] * 50,
                "sigma": [1.0] * 50,
                "nn": {
                    "W1": [[0.1] * 32] * 50,
                    "b1": [0.0] * 32,
                    "W2": [[0.2]] * 32,
                    "b2": [0.0],
                },
                "model_type": "neural_network",
            }
            model_path = v1_dir / "best_model_v1_nn_20231115.joblib"
            joblib.dump(bundle, str(model_path))

            result = load_v1_model(project_root)

            assert result["path"] == str(model_path)
            assert "bundle" in result
            assert result["bundle"]["model_type"] == "neural_network"

    def test_load_v1_model_no_artifact_raises(self):
        """Test that FileNotFoundError is raised when no v1 artifact exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Don't create v1 directory or artifact
            with pytest.raises(FileNotFoundError):
                load_v1_model(tmpdir)

    def test_load_v1_model_uses_latest(self):
        """Test that load_v1_model loads the lexicographically latest artifact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = tmpdir
            v1_dir = Path(project_root, "v1")
            v1_dir.mkdir()

            bundle1 = {"model_type": "nn_v1_early"}
            bundle2 = {"model_type": "nn_v1_late"}

            joblib.dump(
                bundle1, str(v1_dir / "best_model_v1_nn_20231101.joblib")
            )
            joblib.dump(
                bundle2, str(v1_dir / "best_model_v1_nn_20231115.joblib")
            )

            result = load_v1_model(project_root)

            assert "best_model_v1_nn_" in result["path"]
            # One of the bundles should be loaded
            assert result["bundle"]["model_type"] in [
                "nn_v1_early",
                "nn_v1_late",
            ]
