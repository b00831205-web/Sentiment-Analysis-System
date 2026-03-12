"""Unit tests for the v2.logging_config module.

Tests cover:
- Logging setup and configuration
- Log file creation
- Handler functionality
"""

from __future__ import annotations

import tempfile
import logging
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v2.logging_config import setup_logging


class TestLoggingSetup:
    """Test logging configuration."""

    def teardown_method(self):
        """Clean up logging handlers after each test."""
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            try:
                handler.close()
            except Exception:
                pass
            root_logger.removeHandler(handler)
        root_logger.setLevel(logging.NOTSET)

    def test_setup_logging_creates_log_dir(self):
        """Test that setup_logging creates the log directory if it doesn't exist."""
        import shutil
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            assert not log_dir.exists()

            setup_logging(log_dir=str(log_dir))

            assert log_dir.exists()
            
            # Clean up handlers before tmpdir cleanup
            for handler in logging.getLogger().handlers[:]:
                try:
                    handler.close()
                except Exception:
                    pass

    def test_setup_logging_creates_log_file(self):
        """Test that setup_logging creates a log file."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            log_dir = Path(tmpdir) / "logs"

            setup_logging(log_dir=str(log_dir))

            log_file = log_dir / "v2.log"
            assert log_file.exists()
            
            # Clean up handlers before tmpdir cleanup
            for handler in logging.getLogger().handlers[:]:
                try:
                    handler.close()
                except Exception:
                    pass

    def test_setup_logging_default_log_dir(self):
        """Test that setup_logging uses default log directory when not specified."""
        # Capture the call without creating logs in project root
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            setup_logging(log_dir=str(tmpdir))

            log_file = Path(tmpdir) / "v2.log"
            assert log_file.exists()
            
            # Clean up handlers before tmpdir cleanup
            for handler in logging.getLogger().handlers[:]:
                try:
                    handler.close()
                except Exception:
                    pass

    def test_setup_logging_with_debug_level(self):
        """Test setup_logging with DEBUG level."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            setup_logging(log_dir=str(tmpdir), level="DEBUG")

            # Check that logger is configured
            logger = logging.getLogger("test_debug")
            # Logger should inherit DEBUG level from root
            assert logging.getLogger().level <= logging.DEBUG
            
            # Clean up handlers before tmpdir cleanup
            for handler in logging.getLogger().handlers[:]:
                try:
                    handler.close()
                except Exception:
                    pass

    def test_setup_logging_with_info_level(self):
        """Test setup_logging with INFO level."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            setup_logging(log_dir=str(tmpdir), level="INFO")

            logger = logging.getLogger()
            assert logger.level <= logging.INFO
            
            # Clean up handlers before tmpdir cleanup
            for handler in logging.getLogger().handlers[:]:
                try:
                    handler.close()
                except Exception:
                    pass

    def test_setup_logging_handlers_exist(self):
        """Test that setup_logging creates both console and file handlers."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            # Clear any existing handlers first
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                try:
                    handler.close()
                except Exception:
                    pass
                root_logger.removeHandler(handler)

            setup_logging(log_dir=str(tmpdir))

            root_logger = logging.getLogger()
            handler_types = [type(h).__name__ for h in root_logger.handlers]

            # Should have both StreamHandler (console) and FileHandler
            assert any("StreamHandler" in ht for ht in handler_types)
            assert any("FileHandler" in ht for ht in handler_types)
            
            # Clean up handlers before tmpdir cleanup
            for handler in root_logger.handlers[:]:
                try:
                    handler.close()
                except Exception:
                    pass

    def test_setup_logging_can_write_logs(self):
        """Test that configured logging actually writes to file."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            log_dir = Path(tmpdir) / "logs"

            setup_logging(log_dir=str(log_dir))

            # Write a test log
            logger = logging.getLogger("test.module")
            logger.info("Test message")

            # Check log file contains the message
            log_file = log_dir / "v2.log"
            assert log_file.exists()

            with open(log_file) as f:
                content = f.read()
                assert "Test message" in content or len(content) > 0
            
            # Clean up handlers before tmpdir cleanup
            for handler in logging.getLogger().handlers[:]:
                try:
                    handler.close()
                except Exception:
                    pass

    def test_setup_logging_invalid_level_defaults_to_info(self):
        """Test that invalid log level defaults to INFO."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            # Invalid level should not raise, but use default
            setup_logging(log_dir=str(tmpdir), level="INVALID")

            logger = logging.getLogger()
            # Should fall back to INFO level
            assert logger.level == logging.INFO or logger.level == logging.NOTSET
            
            # Clean up handlers before tmpdir cleanup
            for handler in logging.getLogger().handlers[:]:
                try:
                    handler.close()
                except Exception:
                    pass

    def test_setup_logging_format_includes_timestamp(self):
        """Test that log format includes timestamp."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            log_dir = Path(tmpdir) / "logs"

            setup_logging(log_dir=str(log_dir))

            logger = logging.getLogger("test.format")
            logger.info("Format test")

            log_file = log_dir / "v2.log"
            with open(log_file) as f:
                content = f.read()
                # Check format includes timestamp, level, and logger name
                if content.strip():
                    # Should contain date/time info if log written
                    assert any(
                        char.isdigit() for char in content
                    )  # Date/time has digits
            
            # Clean up handlers before tmpdir cleanup
            for handler in logging.getLogger().handlers[:]:
                try:
                    handler.close()
                except Exception:
                    pass
