"""Tests for the structured logging module."""

import logging

from src.utils.logger import get_logger, log_event


class TestGetLogger:
    """Tests for get_logger function."""

    def test_returns_logger_instance(self) -> None:
        """get_logger returns a logging.Logger instance."""
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)

    def test_logger_has_handler(self) -> None:
        """Logger has at least one handler configured."""
        logger = get_logger("test.handler")
        assert len(logger.handlers) > 0

    def test_logger_name_matches(self) -> None:
        """Logger name matches the requested name."""
        logger = get_logger("test.name")
        assert logger.name == "test.name"

    def test_same_name_returns_same_logger(self) -> None:
        """Requesting the same name returns the cached logger."""
        logger1 = get_logger("test.cached")
        logger2 = get_logger("test.cached")
        assert logger1 is logger2

    def test_custom_level(self) -> None:
        """Logger respects custom level override."""
        logger = get_logger("test.custom_level", level="DEBUG")
        assert logger.level == logging.DEBUG

    def test_default_level_is_info_or_config(self) -> None:
        """Logger defaults to INFO when no config available."""
        logger = get_logger("test.default_level")
        assert logger.level in (logging.DEBUG, logging.INFO)


class TestLogEvent:
    """Tests for log_event structured logging."""

    def test_log_event_info(self, capfd) -> None:
        """log_event outputs info messages."""
        logger = get_logger("test.event_info")
        log_event(logger, "info", "test message", key1="value1")
        captured = capfd.readouterr()
        assert "test message" in captured.out
        assert "key1=value1" in captured.out

    def test_log_event_without_kwargs(self, capfd) -> None:
        """log_event works without extra kwargs."""
        logger = get_logger("test.event_no_kwargs")
        log_event(logger, "info", "plain message")
        captured = capfd.readouterr()
        assert "plain message" in captured.out

    def test_log_event_multiple_kwargs(self, capfd) -> None:
        """log_event handles multiple key-value pairs."""
        logger = get_logger("test.event_multi")
        log_event(logger, "info", "multi", a="1", b="2")
        captured = capfd.readouterr()
        assert "a=1" in captured.out
        assert "b=2" in captured.out
