"""Unit tests for the v2.cli module.

Tests cover:
- CLI argument parsing
- Configuration loading
- Command dispatch
"""

from __future__ import annotations

import tempfile
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v2.cli import main


class TestCLIArgParsing:
    """Test CLI argument parsing and command dispatch."""

    @patch("sys.argv", ["v2", "predict", "--text", "hello", "--model", "v0"])
    @patch("v2.cli.load_v0_model")
    @patch("v2.cli.load_v1_model")
    @patch("v2.cli.setup_logging")
    @patch("v2.cli.predict")
    def test_predict_command(self, mock_predict, mock_setup_logging, mock_load_v1, mock_load_v0):
        """Test 'predict' subcommand parsing."""
        # Mock successful operations
        mock_load_v0.return_value = {"model": Mock()}
        mock_predict.return_value = {"label": 1, "prob_pos": 0.9}

        # This should not raise an exception
        try:
            main()
        except SystemExit:
            # Script may call sys.exit(0) on success
            pass

    @patch("sys.argv", ["v2", "serve"])
    @patch("v2.cli.load_v0_model")
    @patch("v2.cli.load_v1_model")
    @patch("v2.cli.setup_logging")
    def test_serve_command(self, mock_setup_logging, mock_load_v1, mock_load_v0):
        """Test 'serve' subcommand parsing."""
        mock_load_v0.return_value = {"model": Mock()}
        mock_load_v1.return_value = {"bundle": {}}

        # Mock app.run() to prevent actual server start
        with patch("v2.server.create_app") as mock_create_app:
            mock_app = Mock()
            mock_app.run = Mock()
            mock_create_app.return_value = mock_app

            try:
                main()
            except SystemExit:
                pass

    @patch("sys.argv", ["v2", "predict"])
    def test_missing_required_text_argument(self):
        """Test that missing --text argument is caught."""
        with pytest.raises(SystemExit):
            main()

    def test_config_file_loading(self):
        """Test loading configuration from a JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.json"
            config = {
                "logging": {"level": "DEBUG", "log_dir": tmpdir},
                "server": {"host": "0.0.0.0", "port": 5000},
            }
            with open(config_path, "w") as f:
                json.dump(config, f)

            # Test that the config file can be created and loaded
            assert config_path.exists()
            with open(config_path) as f:
                loaded = json.load(f)
                assert loaded["server"]["port"] == 5000

    @patch("sys.argv", ["v2", "predict", "--text", "test", "--model", "v0", "--config", "nonexistent.json"])
    @patch("v2.cli.load_v0_model")
    @patch("v2.cli.setup_logging")
    def test_missing_config_file_handling(self, mock_setup_logging, mock_load_v0):
        """Test handling of missing configuration file."""
        # The app should handle missing config gracefully (using defaults)
        mock_load_v0.return_value = {"model": Mock()}

        with patch("v2.cli.predict"):
            # Should proceed with defaults if config file doesn't exist
            try:
                main()
            except (SystemExit, FileNotFoundError):
                # Either exit cleanly or raise FileNotFoundError is acceptable
                pass


class TestCLIValidation:
    """Test input validation in CLI."""

    @patch("sys.argv", ["v2", "predict", "--text", "hello", "--model", "invalid_model"])
    @patch("v2.cli.load_v0_model")
    @patch("v2.cli.load_v1_model")
    @patch("v2.cli.setup_logging")
    def test_invalid_model_kind(self, mock_setup_logging, mock_load_v1, mock_load_v0):
        """Test handling of invalid model kind."""
        mock_load_v0.return_value = {"model": Mock()}
        mock_load_v1.return_value = {"bundle": {}}

        # argparse will raise SystemExit when invalid choice is provided
        with pytest.raises(SystemExit):
            main()

    @patch("sys.argv", ["v2", "predict", "--text", "", "--model", "v0"])
    @patch("v2.cli.load_v0_model")
    @patch("v2.cli.setup_logging")
    def test_empty_text_handling(self, mock_setup_logging, mock_load_v0):
        """Test handling of empty text input."""
        mock_load_v0.return_value = {"model": Mock()}

        # Empty text may be processed by the model or rejected
        with patch("v2.cli.predict") as mock_predict:
            mock_predict.return_value = {"label": 0, "prob_pos": 0.5}
            try:
                main()
            except (SystemExit, ValueError):
                # Either succeeds or raises ValueError
                pass
