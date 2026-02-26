"""Shared test fixtures for the ML deployment platform."""

import os
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def sample_config() -> dict:
    """Return a minimal test configuration dictionary."""
    return {
        "mlflow": {
            "tracking_uri": "sqlite:///test_mlflow.db",
            "artifact_root": "./test_mlruns",
            "experiment_name": "test-experiment",
            "registry_uri": "sqlite:///test_mlflow.db",
        },
        "training": {
            "dataset": "iris",
            "test_size": 0.2,
            "random_state": 42,
            "models": {
                "random_forest": {
                    "n_estimators": 10,
                    "max_depth": 5,
                    "min_samples_split": 2,
                },
                "xgboost": {
                    "n_estimators": 10,
                    "max_depth": 3,
                    "learning_rate": 0.1,
                },
            },
        },
        "serving": {
            "host": "127.0.0.1",
            "port": 8000,
            "model_name": "test-model",
            "model_stage": "Production",
            "reload_interval_seconds": 60,
            "log_predictions": False,
        },
        "canary": {
            "enabled": False,
            "traffic_percentage": 10,
            "promotion_step": 10,
            "rollback_threshold": 0.05,
            "min_requests_before_eval": 50,
        },
        "monitoring": {
            "drift": {
                "accuracy_threshold": 0.85,
                "check_interval_seconds": 300,
                "min_samples": 100,
                "window_size": 500,
            },
        },
        "retraining": {
            "auto_retrain": True,
            "accuracy_drop_threshold": 0.05,
            "min_samples_for_retrain": 200,
            "max_retrain_frequency_seconds": 3600,
        },
        "logging": {
            "level": "DEBUG",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": None,
        },
    }


@pytest.fixture
def config_file(sample_config: dict, tmp_path: Path) -> Path:
    """Create a temporary config YAML file and return its path."""
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config, f)
    return config_path


@pytest.fixture(autouse=True)
def _reset_config_cache() -> None:
    """Reset the config cache before each test."""
    from src.utils.config import reset_config_cache

    reset_config_cache()


@pytest.fixture(autouse=True)
def _clean_env_vars():
    """Remove MLP_ environment variables after each test."""
    yield
    for key in list(os.environ.keys()):
        if key.startswith("MLP_"):
            del os.environ[key]
