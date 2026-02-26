"""Tests for the training pipeline and evaluation modules."""

import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.training.evaluate import compare_metrics, evaluate_model
from src.training.train import build_model, load_dataset, train_model


class TestLoadDataset:
    """Tests for dataset loading."""

    def test_load_iris(self) -> None:
        """Loading iris dataset returns correct shapes."""
        X_train, X_test, y_train, y_test, features = load_dataset("iris")
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert X_train.shape[1] == 4
        assert len(features) == 4

    def test_load_wine(self) -> None:
        """Loading wine dataset returns correct shapes."""
        X_train, X_test, y_train, y_test, features = load_dataset("wine")
        assert X_train.shape[0] > 0
        assert X_test.shape[1] == 13

    def test_unknown_dataset_raises(self) -> None:
        """Unknown dataset name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("nonexistent")

    def test_test_size_split(self) -> None:
        """Test size parameter controls the split ratio."""
        X_train, X_test, _, _, _ = load_dataset("iris", test_size=0.3)
        total = X_train.shape[0] + X_test.shape[0]
        assert abs(X_test.shape[0] / total - 0.3) < 0.05

    def test_random_state_reproducibility(self) -> None:
        """Same random state produces identical splits."""
        result1 = load_dataset("iris", random_state=42)
        result2 = load_dataset("iris", random_state=42)
        np.testing.assert_array_equal(result1[0], result2[0])


class TestBuildModel:
    """Tests for model building."""

    def test_build_random_forest(self) -> None:
        """Building a random forest returns the correct type."""
        model = build_model("random_forest", {"n_estimators": 10, "max_depth": 3})
        assert isinstance(model, RandomForestClassifier)

    def test_build_xgboost(self) -> None:
        """Building an xgboost model succeeds."""
        from xgboost import XGBClassifier

        model = build_model("xgboost", {"n_estimators": 10, "max_depth": 3})
        assert isinstance(model, XGBClassifier)

    def test_build_unknown_raises(self) -> None:
        """Unknown model type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model type"):
            build_model("unknown_model", {})


class TestEvaluateModel:
    """Tests for model evaluation."""

    def test_evaluate_returns_all_metrics(self) -> None:
        """Evaluation returns accuracy, precision, recall, f1, confusion_matrix."""
        X_train, X_test, y_train, y_test, _ = load_dataset("iris")
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "confusion_matrix" in metrics

    def test_metrics_are_valid_ranges(self) -> None:
        """All scalar metrics are between 0 and 1."""
        X_train, X_test, y_train, y_test, _ = load_dataset("iris")
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)

        for key in ["accuracy", "precision", "recall", "f1"]:
            assert 0 <= metrics[key] <= 1

    def test_confusion_matrix_shape(self) -> None:
        """Confusion matrix has correct dimensions for iris (3 classes)."""
        X_train, X_test, y_train, y_test, _ = load_dataset("iris")
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)
        cm = metrics["confusion_matrix"]
        assert len(cm) == 3
        assert len(cm[0]) == 3


class TestCompareMetrics:
    """Tests for metric comparison."""

    def test_compare_returns_deltas(self) -> None:
        """Comparison returns delta values for each metric."""
        metrics_a = {"accuracy": 0.8, "precision": 0.75, "recall": 0.7, "f1": 0.72}
        metrics_b = {"accuracy": 0.9, "precision": 0.85, "recall": 0.8, "f1": 0.82}

        deltas = compare_metrics(metrics_a, metrics_b)

        assert abs(deltas["accuracy_delta"] - 0.1) < 1e-6
        assert abs(deltas["precision_delta"] - 0.1) < 1e-6

    def test_compare_negative_deltas(self) -> None:
        """Negative deltas indicate regression."""
        metrics_a = {"accuracy": 0.9, "f1": 0.85}
        metrics_b = {"accuracy": 0.8, "f1": 0.75}

        deltas = compare_metrics(metrics_a, metrics_b)
        assert deltas["accuracy_delta"] < 0


class TestTrainModel:
    """Integration tests for the full training pipeline."""

    def test_train_random_forest(self, config_file: Path) -> None:
        """Training a random forest returns valid results with MLflow logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking_uri = f"sqlite:///{tmpdir}/mlflow.db"

            result = train_model(
                model_type="random_forest",
                config_path=str(config_file),
                tracking_uri=tracking_uri,
            )

            assert "run_id" in result
            assert result["model_type"] == "random_forest"
            assert 0 <= result["metrics"]["accuracy"] <= 1
            assert result["training_duration"] > 0

    def test_train_xgboost(self, config_file: Path) -> None:
        """Training an XGBoost model returns valid results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking_uri = f"sqlite:///{tmpdir}/mlflow.db"

            result = train_model(
                model_type="xgboost",
                config_path=str(config_file),
                tracking_uri=tracking_uri,
            )

            assert result["model_type"] == "xgboost"
            assert 0 <= result["metrics"]["accuracy"] <= 1

    def test_mlflow_run_logged(self, config_file: Path) -> None:
        """MLflow run is properly logged with metrics and params."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking_uri = f"sqlite:///{tmpdir}/mlflow.db"

            result = train_model(
                model_type="random_forest",
                config_path=str(config_file),
                tracking_uri=tracking_uri,
            )

            mlflow.set_tracking_uri(tracking_uri)
            run = mlflow.get_run(result["run_id"])

            assert run.data.metrics["accuracy"] > 0
            assert "n_estimators" in run.data.params
            assert run.data.tags["model_type"] == "random_forest"
