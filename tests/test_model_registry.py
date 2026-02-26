"""Tests for MLflow Model Registry integration."""

import tempfile
from pathlib import Path

import pytest

from src.serving.model_loader import (
    compare_model_versions,
    get_model_info,
    load_model_by_stage,
    register_model,
    transition_model_stage,
)
from src.training.train import train_model


@pytest.fixture
def mlflow_setup(config_file: Path):
    """Set up MLflow with a trained model and return context."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracking_uri = f"sqlite:///{tmpdir}/mlflow.db"

        result = train_model(
            model_type="random_forest",
            config_path=str(config_file),
            tracking_uri=tracking_uri,
        )

        yield {
            "tracking_uri": tracking_uri,
            "run_id": result["run_id"],
            "model_name": "test-registry-model",
            "config_path": str(config_file),
            "tmpdir": tmpdir,
        }


class TestRegisterModel:
    """Tests for model registration."""

    def test_register_model_success(self, mlflow_setup: dict) -> None:
        """Registering a model returns a valid ModelVersion."""
        mv = register_model(
            run_id=mlflow_setup["run_id"],
            model_name=mlflow_setup["model_name"],
            tracking_uri=mlflow_setup["tracking_uri"],
        )
        assert mv.version is not None
        assert mv.name == mlflow_setup["model_name"]

    def test_register_model_with_description(self, mlflow_setup: dict) -> None:
        """Registering with description sets the model description."""
        mv = register_model(
            run_id=mlflow_setup["run_id"],
            model_name=mlflow_setup["model_name"],
            tracking_uri=mlflow_setup["tracking_uri"],
            description="Test model for registry",
        )
        assert mv.version is not None

    def test_register_model_with_tags(self, mlflow_setup: dict) -> None:
        """Registering with tags sets version tags."""
        mv = register_model(
            run_id=mlflow_setup["run_id"],
            model_name=mlflow_setup["model_name"],
            tracking_uri=mlflow_setup["tracking_uri"],
            tags={"env": "test", "team": "ml"},
        )
        assert mv.version is not None

    def test_register_multiple_versions(self, mlflow_setup: dict) -> None:
        """Registering the same run twice creates two versions."""
        mv1 = register_model(
            run_id=mlflow_setup["run_id"],
            model_name=mlflow_setup["model_name"],
            tracking_uri=mlflow_setup["tracking_uri"],
        )
        mv2 = register_model(
            run_id=mlflow_setup["run_id"],
            model_name=mlflow_setup["model_name"],
            tracking_uri=mlflow_setup["tracking_uri"],
        )
        assert int(mv2.version) > int(mv1.version)


class TestTransitionStage:
    """Tests for stage transitions."""

    def test_transition_to_staging(self, mlflow_setup: dict) -> None:
        """Transitioning to staging sets the alias."""
        mv = register_model(
            run_id=mlflow_setup["run_id"],
            model_name=mlflow_setup["model_name"],
            tracking_uri=mlflow_setup["tracking_uri"],
        )
        result = transition_model_stage(
            model_name=mlflow_setup["model_name"],
            version=mv.version,
            stage="staging",
            tracking_uri=mlflow_setup["tracking_uri"],
        )
        assert result is not None

    def test_transition_to_production(self, mlflow_setup: dict) -> None:
        """Transitioning to production sets the alias."""
        mv = register_model(
            run_id=mlflow_setup["run_id"],
            model_name=mlflow_setup["model_name"],
            tracking_uri=mlflow_setup["tracking_uri"],
        )
        result = transition_model_stage(
            model_name=mlflow_setup["model_name"],
            version=mv.version,
            stage="production",
            tracking_uri=mlflow_setup["tracking_uri"],
        )
        assert result is not None

    def test_transition_invalid_stage_raises(self, mlflow_setup: dict) -> None:
        """Invalid stage name raises ValueError."""
        mv = register_model(
            run_id=mlflow_setup["run_id"],
            model_name=mlflow_setup["model_name"],
            tracking_uri=mlflow_setup["tracking_uri"],
        )
        with pytest.raises(ValueError, match="Invalid stage"):
            transition_model_stage(
                model_name=mlflow_setup["model_name"],
                version=mv.version,
                stage="invalid",
                tracking_uri=mlflow_setup["tracking_uri"],
            )

    def test_transition_to_archived(self, mlflow_setup: dict) -> None:
        """Archiving removes active aliases from the version."""
        mv = register_model(
            run_id=mlflow_setup["run_id"],
            model_name=mlflow_setup["model_name"],
            tracking_uri=mlflow_setup["tracking_uri"],
        )
        transition_model_stage(
            model_name=mlflow_setup["model_name"],
            version=mv.version,
            stage="production",
            tracking_uri=mlflow_setup["tracking_uri"],
        )
        result = transition_model_stage(
            model_name=mlflow_setup["model_name"],
            version=mv.version,
            stage="archived",
            tracking_uri=mlflow_setup["tracking_uri"],
        )
        assert result is not None


class TestLoadModelByStage:
    """Tests for loading models by stage."""

    def test_load_production_model(self, mlflow_setup: dict) -> None:
        """Loading a production model returns a usable model."""
        mv = register_model(
            run_id=mlflow_setup["run_id"],
            model_name=mlflow_setup["model_name"],
            tracking_uri=mlflow_setup["tracking_uri"],
        )
        transition_model_stage(
            model_name=mlflow_setup["model_name"],
            version=mv.version,
            stage="production",
            tracking_uri=mlflow_setup["tracking_uri"],
        )
        model = load_model_by_stage(
            model_name=mlflow_setup["model_name"],
            stage="production",
            tracking_uri=mlflow_setup["tracking_uri"],
        )
        assert model is not None

    def test_load_missing_stage_raises(self, mlflow_setup: dict) -> None:
        """Loading from a stage with no model raises ValueError."""
        register_model(
            run_id=mlflow_setup["run_id"],
            model_name=mlflow_setup["model_name"],
            tracking_uri=mlflow_setup["tracking_uri"],
        )
        with pytest.raises(ValueError, match="No model found"):
            load_model_by_stage(
                model_name=mlflow_setup["model_name"],
                stage="production",
                tracking_uri=mlflow_setup["tracking_uri"],
            )


class TestGetModelInfo:
    """Tests for model info retrieval."""

    def test_get_info_returns_versions(self, mlflow_setup: dict) -> None:
        """Model info includes version list."""
        register_model(
            run_id=mlflow_setup["run_id"],
            model_name=mlflow_setup["model_name"],
            tracking_uri=mlflow_setup["tracking_uri"],
        )
        info = get_model_info(
            model_name=mlflow_setup["model_name"],
            tracking_uri=mlflow_setup["tracking_uri"],
        )
        assert info["name"] == mlflow_setup["model_name"]
        assert len(info["versions"]) >= 1

    def test_get_info_nonexistent_raises(self, mlflow_setup: dict) -> None:
        """Querying a non-existent model raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            get_model_info(
                model_name="nonexistent-model",
                tracking_uri=mlflow_setup["tracking_uri"],
            )


class TestCompareVersions:
    """Tests for model version comparison."""

    def test_compare_two_versions(self, mlflow_setup: dict) -> None:
        """Comparing two versions returns metrics and deltas."""
        mv1 = register_model(
            run_id=mlflow_setup["run_id"],
            model_name=mlflow_setup["model_name"],
            tracking_uri=mlflow_setup["tracking_uri"],
        )

        result2 = train_model(
            model_type="random_forest",
            config_path=mlflow_setup["config_path"],
            tracking_uri=mlflow_setup["tracking_uri"],
        )
        mv2 = register_model(
            run_id=result2["run_id"],
            model_name=mlflow_setup["model_name"],
            tracking_uri=mlflow_setup["tracking_uri"],
        )

        comparison = compare_model_versions(
            model_name=mlflow_setup["model_name"],
            version_a=mv1.version,
            version_b=mv2.version,
            tracking_uri=mlflow_setup["tracking_uri"],
        )

        assert "version_a" in comparison
        assert "version_b" in comparison
        assert "deltas" in comparison
        assert "accuracy" in comparison["deltas"]
