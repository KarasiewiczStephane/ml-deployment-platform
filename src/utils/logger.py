"""Structured logging setup for the ML deployment platform.

Provides a configured logger with consistent formatting across all modules.
"""

import logging
import sys
from typing import Any


_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str, level: str | None = None) -> logging.Logger:
    """Get or create a configured logger.

    Args:
        name: Logger name, typically __name__ of the calling module.
        level: Optional override for log level. If None, uses config or defaults to INFO.

    Returns:
        Configured logging.Logger instance.
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)

    if not logger.handlers:
        log_level = _resolve_level(level)
        logger.setLevel(log_level)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)

        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        logger.propagate = False

    _loggers[name] = logger
    return logger


def _resolve_level(level: str | None) -> int:
    """Resolve log level from explicit value, config, or default.

    Args:
        level: Explicit level string, or None to use config/default.

    Returns:
        Numeric logging level.
    """
    if level is not None:
        return getattr(logging, level.upper(), logging.INFO)

    try:
        from src.utils.config import get_config_value

        config_level = get_config_value("logging.level", "INFO")
        return getattr(logging, str(config_level).upper(), logging.INFO)
    except Exception:
        return logging.INFO


def log_event(
    logger: logging.Logger,
    level: str,
    message: str,
    **kwargs: Any,
) -> None:
    """Log a structured event with additional context fields.

    Args:
        logger: Logger instance to use.
        level: Log level string (e.g., 'info', 'error').
        message: Log message.
        **kwargs: Additional key-value pairs to include in the log message.
    """
    extra_info = " | ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    full_message = f"{message} | {extra_info}" if extra_info else message

    log_method = getattr(logger, level.lower(), logger.info)
    log_method(full_message)
