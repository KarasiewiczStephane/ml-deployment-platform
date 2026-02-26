"""Configuration loader for the ML deployment platform.

Loads settings from configs/config.yaml with environment variable overrides.
"""

import os
from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "configs" / "config.yaml"
)

_config_cache: dict[str, Any] | None = None


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML config file. Defaults to configs/config.yaml.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    global _config_cache

    if _config_cache is not None and config_path is None:
        return _config_cache

    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    config = _apply_env_overrides(config)
    _config_cache = config

    return config


def get_config_value(key_path: str, default: Any = None) -> Any:
    """Get a nested config value using dot notation.

    Args:
        key_path: Dot-separated path to the config key (e.g., 'mlflow.tracking_uri').
        default: Default value if the key is not found.

    Returns:
        The configuration value at the specified path, or the default.
    """
    config = load_config()
    keys = key_path.split(".")
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value


def reset_config_cache() -> None:
    """Clear the cached configuration to force a reload on next access."""
    global _config_cache
    _config_cache = None


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Apply environment variable overrides to config values.

    Environment variables follow the pattern: MLP_{SECTION}_{KEY} (uppercase).
    For example, MLP_MLFLOW_TRACKING_URI overrides mlflow.tracking_uri.

    Args:
        config: The parsed configuration dictionary.

    Returns:
        Configuration with environment overrides applied.
    """
    env_mappings = {
        "MLP_MLFLOW_TRACKING_URI": ("mlflow", "tracking_uri"),
        "MLP_MLFLOW_EXPERIMENT_NAME": ("mlflow", "experiment_name"),
        "MLP_SERVING_HOST": ("serving", "host"),
        "MLP_SERVING_PORT": ("serving", "port"),
        "MLP_SERVING_MODEL_NAME": ("serving", "model_name"),
        "MLP_SERVING_MODEL_STAGE": ("serving", "model_stage"),
        "MLP_LOGGING_LEVEL": ("logging", "level"),
    }

    for env_var, key_path in env_mappings.items():
        env_value = os.environ.get(env_var)
        if env_value is not None:
            section, key = key_path
            if section in config and isinstance(config[section], dict):
                original = config[section].get(key)
                if isinstance(original, int):
                    config[section][key] = int(env_value)
                elif isinstance(original, float):
                    config[section][key] = float(env_value)
                elif isinstance(original, bool):
                    config[section][key] = env_value.lower() in ("true", "1", "yes")
                else:
                    config[section][key] = env_value

    return config
