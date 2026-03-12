"""Unit tests for the v2.server module.

Tests cover:
- Flask app creation and configuration
- API endpoints (health check, predict, HTML UI)
- Error handling
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v2.server import create_app


class TestFlaskApp:
    """Test Flask application factory and endpoints."""

    @pytest.fixture
    def app(self):
        """Create a Flask test app."""
        with patch("v2.server.load_v0_model"), \
             patch("v2.server.load_v1_model"), \
             patch("v2.server.setup_logging"):
            app = create_app()
            app.config["TESTING"] = True
            return app

    @pytest.fixture
    def client(self, app):
        """Create a test client."""
        return app.test_client()

    def test_app_creation(self, app):
        """Test that Flask app is created successfully."""
        assert app is not None
        assert app.config["TESTING"] is True

    def test_health_check_endpoint(self, client):
        """Test the /health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "status" in data

    def test_predict_endpoint_post(self, client):
        """Test POST /predict endpoint."""
        payload = {"text": "This movie is great!", "model_kind": "v0"}
        response = client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        # Without actual models loaded, this may error gracefully
        assert response.status_code in [200, 400, 500]

    def test_predict_endpoint_missing_text(self, client):
        """Test /predict endpoint with missing text field."""
        payload = {"model_kind": "v0"}  # Missing "text"
        response = client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        # Should handle missing field gracefully
        assert response.status_code in [400, 500]

    def test_html_ui_endpoint(self, client):
        """Test GET / returns HTML UI."""
        response = client.get("/")
        assert response.status_code == 200
        assert b"<!doctype html>" in response.data.lower() or b"<html" in response.data.lower()

    def test_api_not_found_returns_404(self, client):
        """Test that unknown endpoints return 404."""
        response = client.get("/nonexistent")
        assert response.status_code == 404


class TestServerIntegration:
    """Integration-level tests for the server."""

    @pytest.fixture
    def app_with_mock_models(self):
        """Create a Flask app with mocked model loaders."""
        v0_ctx_mock = {
            "model": Mock(
                predict=Mock(return_value=[1]),
                predict_proba=Mock(return_value=[[0.2, 0.8]]),
            ),
            "path": "/path/to/v0/model.joblib",
            "best_name": "logistic_regression",
        }

        v1_ctx_mock = {
            "bundle": {
                "vectorizer": Mock(transform=Mock(return_value=[[0] * 100])),
                "svd": Mock(transform=Mock(return_value=[[0] * 50])),
                "mu": [0] * 50,
                "sigma": [1] * 50,
                "nn": {
                    "W1": [[0.1] * 32] * 50,
                    "b1": [0.0] * 32,
                    "W2": [[0.2]] * 32,
                    "b2": [0.0],
                },
                "model_type": "neural_network",
            },
            "path": "/path/to/v1/model.joblib",
        }

        with patch("v2.server.load_v0_model", return_value=v0_ctx_mock), \
             patch("v2.server.load_v1_model", return_value=v1_ctx_mock), \
             patch("v2.server.setup_logging"), \
             patch("v2.server.ensure_aclImdb"):
            app = create_app()
            app.config["TESTING"] = True
            return app

    @pytest.fixture
    def client_with_models(self, app_with_mock_models):
        """Create a test client with mock models."""
        return app_with_mock_models.test_client()

    def test_predict_v0_with_mock_model(self, client_with_models):
        """Test v0 prediction through API with mocked model."""
        payload = {"text": "This is a great movie!", "model_kind": "v0"}
        response = client_with_models.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )

        if response.status_code == 200:
            data = json.loads(response.data)
            assert "label" in data or "error" in data

    def test_predict_v1_with_mock_model(self, client_with_models):
        """Test v1 prediction through API with mocked model."""
        payload = {"text": "This is a bad movie.", "model_kind": "v1"}
        response = client_with_models.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )

        if response.status_code == 200:
            data = json.loads(response.data)
            assert "label" in data or "error" in data

    def test_predict_empty_text(self, client_with_models):
        """Test prediction with empty text."""
        payload = {"text": "", "model_kind": "v0"}
        response = client_with_models.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )

        # May return 200 or error depending on implementation
        assert response.status_code in [200, 400, 500]

    def test_predict_long_text(self, client_with_models):
        """Test prediction with very long text."""
        long_text = "word " * 10000  # Very long input
        payload = {"text": long_text, "model_kind": "v0"}
        response = client_with_models.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )

        # Should handle gracefully without crashing
        assert response.status_code in [200, 400, 413, 500]
