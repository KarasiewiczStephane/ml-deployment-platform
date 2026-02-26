"""Tests for configuration loading and environment overrides."""

import os
from pathlib import Path

import pytest

from src.utils.config import get_config_value, load_config, reset_config_cache


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_from_file(self, config_file: Path) -> None:
        """Loading a valid config file returns a populated dictionary."""
        config = load_config(config_path=config_file)
        assert isinstance(config, dict)
        assert "mlflow" in config
        assert "training" in config

    def test_load_config_missing_file_raises(self, tmp_path: Path) -> None:
        """Loading a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config(config_path=tmp_path / "nonexistent.yaml")

    def test_load_config_caching(self, config_file: Path) -> None:
        """Default config is cached after first load."""
        config1 = load_config(config_path=config_file)
        reset_config_cache()
        config2 = load_config(config_path=config_file)
        assert config1 == config2

    def test_load_config_values(self, config_file: Path) -> None:
        """Config values match what was written to the file."""
        config = load_config(config_path=config_file)
        assert config["training"]["dataset"] == "iris"
        assert config["training"]["test_size"] == 0.2
        assert config["training"]["random_state"] == 42

    def test_mlflow_config_values(self, config_file: Path) -> None:
        """MLflow configuration section is properly loaded."""
        config = load_config(config_path=config_file)
        assert config["mlflow"]["tracking_uri"] == "sqlite:///test_mlflow.db"
        assert config["mlflow"]["experiment_name"] == "test-experiment"

    def test_serving_config_values(self, config_file: Path) -> None:
        """Serving configuration section is properly loaded."""
        config = load_config(config_path=config_file)
        assert config["serving"]["port"] == 8000
        assert config["serving"]["model_name"] == "test-model"


class TestGetConfigValue:
    """Tests for get_config_value with dot notation."""

    def test_get_nested_value(self, config_file: Path) -> None:
        """Dot notation retrieves nested config values."""
        load_config(config_path=config_file)
        value = get_config_value("mlflow.tracking_uri")
        assert value == "sqlite:///test_mlflow.db"

    def test_get_deeply_nested_value(self, config_file: Path) -> None:
        """Dot notation works for deeply nested paths."""
        load_config(config_path=config_file)
        value = get_config_value("training.models.random_forest.n_estimators")
        assert value == 10

    def test_get_missing_key_returns_default(self, config_file: Path) -> None:
        """Missing keys return the provided default value."""
        load_config(config_path=config_file)
        value = get_config_value("nonexistent.key", default="fallback")
        assert value == "fallback"

    def test_get_missing_key_returns_none(self, config_file: Path) -> None:
        """Missing keys without default return None."""
        load_config(config_path=config_file)
        value = get_config_value("nonexistent.key")
        assert value is None


class TestEnvOverrides:
    """Tests for environment variable overrides."""

    def test_string_override(self, config_file: Path) -> None:
        """String env var overrides config value."""
        os.environ["MLP_MLFLOW_TRACKING_URI"] = "http://remote-mlflow:5000"
        config = load_config(config_path=config_file)
        assert config["mlflow"]["tracking_uri"] == "http://remote-mlflow:5000"

    def test_int_override(self, config_file: Path) -> None:
        """Integer env var is properly converted."""
        os.environ["MLP_SERVING_PORT"] = "9090"
        config = load_config(config_path=config_file)
        assert config["serving"]["port"] == 9090

    def test_unset_env_var_no_effect(self, config_file: Path) -> None:
        """Config values remain unchanged without env vars set."""
        config = load_config(config_path=config_file)
        assert config["serving"]["port"] == 8000


class TestResetCache:
    """Tests for config cache reset."""

    def test_reset_forces_reload(self, config_file: Path) -> None:
        """Resetting cache causes next load to re-read from file."""
        config1 = load_config(config_path=config_file)
        reset_config_cache()
        config2 = load_config(config_path=config_file)
        assert config1 == config2
