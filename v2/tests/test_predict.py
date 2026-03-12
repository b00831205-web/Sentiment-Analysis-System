"""Unit tests for the v2.predict module.

Tests cover:
- sigmoid function numerical stability
- v0 model prediction interface
- v1 model prediction interface
- main dispatch function
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v2.predict import _sigmoid, predict_v0, predict_v1, predict


class TestSigmoid:
    """Test the numerically stable sigmoid implementation."""

    def test_sigmoid_zero_returns_half(self):
        """Sigmoid of 0 should be 0.5."""
        result = _sigmoid(np.array([0.0]))
        assert np.isclose(result[0], 0.5)

    def test_sigmoid_large_positive_approaches_one(self):
        """Sigmoid of large positive values should approach 1.0."""
        result = _sigmoid(np.array([100.0]))
        assert np.isclose(result[0], 1.0, atol=1e-6)

    def test_sigmoid_large_negative_approaches_zero(self):
        """Sigmoid of large negative values should approach 0.0."""
        result = _sigmoid(np.array([-100.0]))
        assert np.isclose(result[0], 0.0, atol=1e-6)

    def test_sigmoid_clips_extreme_values(self):
        """Sigmoid should handle extreme values without overflow."""
        # Values beyond [-30, 30] are clipped
        result = _sigmoid(np.array([1000.0, -1000.0]))
        assert len(result) == 2
        assert np.all(np.isfinite(result))

    def test_sigmoid_array_input(self):
        """Sigmoid should handle array inputs element-wise."""
        x = np.array([-1.0, 0.0, 1.0])
        result = _sigmoid(x)
        assert len(result) == 3
        # Check monotonicity: results should be increasing
        assert result[0] < result[1] < result[2]


class TestPredictV0:
    """Test v0 model prediction interface."""

    def test_predict_v0_basic(self):
        """Test basic v0 prediction with mock model."""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])

        result = predict_v0(mock_model, "test text")

        assert result["label"] == 1
        assert np.isclose(result["prob_pos"], 0.7)
        mock_model.predict.assert_called_once_with(["test text"])
        mock_model.predict_proba.assert_called_once_with(["test text"])

    def test_predict_v0_without_proba(self):
        """Test v0 prediction when model lacks predict_proba."""
        mock_model = Mock(spec=["predict"])
        mock_model.predict.return_value = np.array([0])

        result = predict_v0(mock_model, "bad text")

        assert result["label"] == 0
        assert result["prob_pos"] is None

    def test_predict_v0_handles_both_classes(self):
        """Test v0 prediction handles both positive and negative predictions."""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0])
        mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])

        result = predict_v0(mock_model, "negative text")

        assert result["label"] == 0
        assert np.isclose(result["prob_pos"], 0.2)


class TestPredictV1:
    """Test v1 model prediction interface."""

    def test_predict_v1_basic(self):
        """Test basic v1 prediction with mock bundle."""
        # Create mock vectorizer, SVD, and NN weights
        mock_vec = Mock()
        mock_vec.transform.return_value = np.random.randn(1, 100)

        mock_svd = Mock()
        mock_svd.transform.return_value = np.random.randn(1, 50)

        bundle = {
            "vectorizer": mock_vec,
            "svd": mock_svd,
            "mu": np.zeros((1, 50)),
            "sigma": np.ones((1, 50)),
            "nn": {
                "W1": np.random.randn(50, 32),
                "b1": np.zeros((1, 32)),
                "W2": np.random.randn(32, 1),
                "b2": np.zeros((1, 1)),
            },
        }

        result = predict_v1(bundle, "test review")

        assert "label" in result
        assert "prob_pos" in result
        assert result["label"] in (0, 1)
        assert 0 <= result["prob_pos"] <= 1

    def test_predict_v1_missing_vectorizer_raises(self):
        """Test that missing vectorizer raises KeyError."""
        incomplete_bundle = {"svd": Mock()}

        with pytest.raises(KeyError):
            predict_v1(incomplete_bundle, "test text")

    def test_predict_v1_missing_nn_weights_raises(self):
        """Test that missing NN weights raises KeyError."""
        incomplete_bundle = {
            "vectorizer": Mock(),
            "svd": Mock(),
            "mu": np.zeros((1, 50)),
            "sigma": np.ones((1, 50)),
            "nn": {},  # Missing W1, b1, W2, b2
        }

        with pytest.raises(KeyError):
            predict_v1(incomplete_bundle, "test text")


class TestPredictDispatch:
    """Test the main dispatch function."""

    def test_predict_dispatch_v0(self):
        """Test dispatch to v0 model."""
        mock_v0_ctx = {
            "model": Mock(predict=Mock(return_value=np.array([1]))),
            "path": "/path/to/v0/model.joblib",
            "best_name": "logistic_regression",
        }
        mock_v0_ctx["model"].predict_proba = Mock(
            return_value=np.array([[0.2, 0.8]])
        )

        result = predict("v0", mock_v0_ctx, None, "great movie")

        assert result["model"] == "v0"
        assert result["label"] == 1
        assert result["model_path"] == "/path/to/v0/model.joblib"

    def test_predict_dispatch_v1(self):
        """Test dispatch to v1 model."""
        mock_v1_ctx = {
            "bundle": {
                "vectorizer": Mock(transform=Mock(return_value=np.random.randn(1, 100))),
                "svd": Mock(transform=Mock(return_value=np.random.randn(1, 50))),
                "mu": np.zeros((1, 50)),
                "sigma": np.ones((1, 50)),
                "nn": {
                    "W1": np.random.randn(50, 32),
                    "b1": np.zeros((1, 32)),
                    "W2": np.random.randn(32, 1),
                    "b2": np.zeros((1, 1)),
                },
                "model_type": "neural_network_v1",
            },
            "path": "/path/to/v1/model.joblib",
        }

        result = predict("v1", None, mock_v1_ctx, "good movie")

        assert result["model"] == "v1"
        assert "label" in result
        assert "prob_pos" in result
        assert result["model_path"] == "/path/to/v1/model.joblib"

    def test_predict_dispatch_case_insensitive(self):
        """Test that model_kind is case-insensitive."""
        mock_v0_ctx = {
            "model": Mock(predict=Mock(return_value=np.array([0]))),
            "path": "/path/v0.joblib",
            "best_name": "svm",
        }
        mock_v0_ctx["model"].predict_proba = Mock(
            return_value=np.array([[0.6, 0.4]])
        )

        result = predict("V0", mock_v0_ctx, None, "text")
        assert result["model"] == "v0"

    def test_predict_dispatch_v0_not_loaded_raises(self):
        """Test that requesting v0 without loading raises ValueError."""
        with pytest.raises(ValueError, match="v0 model not loaded"):
            predict("v0", None, None, "text")

    def test_predict_dispatch_v1_not_loaded_raises(self):
        """Test that requesting v1 without loading raises ValueError."""
        with pytest.raises(ValueError, match="v1 model not loaded"):
            predict("v1", None, None, "text")

    def test_predict_dispatch_invalid_model_kind_raises(self):
        """Test that invalid model_kind raises ValueError."""
        with pytest.raises(ValueError, match="model must be one of"):
            predict("v2", None, None, "text")

    def test_predict_dispatch_whitespace_stripped(self):
        """Test that whitespace is stripped from model_kind."""
        mock_v0_ctx = {
            "model": Mock(predict=Mock(return_value=np.array([1]))),
            "path": "/path/v0.joblib",
            "best_name": "rf",
        }
        mock_v0_ctx["model"].predict_proba = Mock(
            return_value=np.array([[0.1, 0.9]])
        )

        result = predict("  v0  ", mock_v0_ctx, None, "text")
        assert result["model"] == "v0"
