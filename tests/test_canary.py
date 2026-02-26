"""Tests for canary deployment simulation."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.serving.canary import CanaryManager, VersionMetrics
from src.training.train import load_dataset


@pytest.fixture
def iris_models():
    """Return two trained models on iris data."""
    X_train, X_test, y_train, y_test, _ = load_dataset("iris")

    model_a = RandomForestClassifier(n_estimators=5, random_state=42)
    model_a.fit(X_train, y_train)

    model_b = RandomForestClassifier(n_estimators=10, random_state=99)
    model_b.fit(X_train, y_train)

    return model_a, model_b, X_test, y_test


class TestVersionMetrics:
    """Tests for VersionMetrics dataclass."""

    def test_initial_error_rate(self) -> None:
        """Error rate is 0 with no requests."""
        vm = VersionMetrics()
        assert vm.error_rate == 0.0

    def test_error_rate_calculation(self) -> None:
        """Error rate correctly reflects errors / total."""
        vm = VersionMetrics(total_requests=10, total_errors=2)
        assert abs(vm.error_rate - 0.2) < 1e-6

    def test_avg_latency(self) -> None:
        """Average latency is sum / count."""
        vm = VersionMetrics(total_requests=5, latency_sum=0.5)
        assert abs(vm.avg_latency - 0.1) < 1e-6

    def test_avg_accuracy(self) -> None:
        """Average accuracy is sum / count."""
        vm = VersionMetrics(accuracy_sum=2.7, accuracy_count=3)
        assert abs(vm.avg_accuracy - 0.9) < 1e-6

    def test_avg_accuracy_zero_count(self) -> None:
        """Average accuracy is 0 with no samples."""
        vm = VersionMetrics()
        assert vm.avg_accuracy == 0.0


class TestCanaryManager:
    """Tests for CanaryManager."""

    def test_route_to_primary_when_inactive(self, iris_models) -> None:
        """All traffic goes to primary when canary is not active."""
        model_a, _, _, _ = iris_models
        manager = CanaryManager(primary_model=model_a, primary_version="v1")
        model, version = manager.route_request()
        assert model is model_a
        assert version == "v1"

    def test_route_splits_traffic(self, iris_models) -> None:
        """Traffic is split between primary and canary according to percentage."""
        model_a, model_b, _, _ = iris_models
        manager = CanaryManager(
            primary_model=model_a,
            canary_model=model_b,
            canary_percentage=50,
        )

        canary_count = 0
        total = 1000
        for _ in range(total):
            _, version = manager.route_request()
            if version == "canary":
                canary_count += 1

        ratio = canary_count / total
        assert 0.3 < ratio < 0.7, f"Expected ~50% canary, got {ratio * 100:.1f}%"

    def test_predict_returns_results(self, iris_models) -> None:
        """Predict returns valid predictions, version, and latency."""
        model_a, model_b, X_test, _ = iris_models
        manager = CanaryManager(
            primary_model=model_a,
            canary_model=model_b,
            canary_percentage=50,
        )

        preds, version, latency_ms = manager.predict(X_test[:2])
        assert len(preds) == 2
        assert version in ("primary", "canary")
        assert latency_ms >= 0

    def test_predict_records_metrics(self, iris_models) -> None:
        """Predictions accumulate in version metrics."""
        model_a, model_b, X_test, _ = iris_models
        manager = CanaryManager(
            primary_model=model_a,
            canary_model=model_b,
            canary_percentage=100,
        )

        manager.predict(X_test[:1])
        assert manager.canary_metrics.total_requests == 1

    def test_predict_no_model_raises(self) -> None:
        """Predict with no model raises RuntimeError."""
        manager = CanaryManager()
        with pytest.raises(RuntimeError, match="No model available"):
            manager.predict(np.array([[1, 2, 3, 4]]))

    def test_record_accuracy(self, iris_models) -> None:
        """Recording accuracy updates version metrics."""
        model_a, _, _, _ = iris_models
        manager = CanaryManager(primary_model=model_a)
        manager.record_accuracy("primary", 0.95)
        manager.record_accuracy("primary", 0.90)
        assert abs(manager.primary_metrics.avg_accuracy - 0.925) < 1e-6


class TestCanaryEvaluation:
    """Tests for canary evaluation and decisions."""

    def test_evaluate_insufficient_requests(self, iris_models) -> None:
        """Evaluation returns 'continue' when requests are below threshold."""
        model_a, model_b, _, _ = iris_models
        manager = CanaryManager(
            primary_model=model_a,
            canary_model=model_b,
            min_requests_before_eval=100,
        )
        manager.canary_metrics.total_requests = 5
        result = manager.evaluate_canary()
        assert result["decision"] == "continue"

    def test_evaluate_promotes_good_canary(self, iris_models) -> None:
        """Good canary performance leads to promotion decision."""
        model_a, model_b, _, _ = iris_models
        manager = CanaryManager(
            primary_model=model_a,
            canary_model=model_b,
            min_requests_before_eval=10,
            rollback_threshold=0.05,
        )

        manager.primary_metrics.total_requests = 100
        manager.primary_metrics.accuracy_sum = 90.0
        manager.primary_metrics.accuracy_count = 100

        manager.canary_metrics.total_requests = 50
        manager.canary_metrics.accuracy_sum = 46.0
        manager.canary_metrics.accuracy_count = 50

        result = manager.evaluate_canary()
        assert result["decision"] == "promote"

    def test_evaluate_rolls_back_bad_canary(self, iris_models) -> None:
        """Poor canary accuracy leads to rollback."""
        model_a, model_b, _, _ = iris_models
        manager = CanaryManager(
            primary_model=model_a,
            canary_model=model_b,
            min_requests_before_eval=10,
            rollback_threshold=0.05,
        )

        manager.primary_metrics.total_requests = 100
        manager.primary_metrics.accuracy_sum = 95.0
        manager.primary_metrics.accuracy_count = 100

        manager.canary_metrics.total_requests = 50
        manager.canary_metrics.accuracy_sum = 40.0
        manager.canary_metrics.accuracy_count = 50

        result = manager.evaluate_canary()
        assert result["decision"] == "rollback"
        assert manager.rolled_back is True

    def test_evaluate_inactive(self) -> None:
        """Evaluation when inactive returns 'inactive'."""
        manager = CanaryManager()
        result = manager.evaluate_canary()
        assert result["decision"] == "inactive"


class TestCanaryPromotion:
    """Tests for canary promotion and rollback."""

    def test_promote_increases_percentage(self, iris_models) -> None:
        """Promotion increases traffic percentage by step."""
        model_a, model_b, _, _ = iris_models
        manager = CanaryManager(
            primary_model=model_a,
            canary_model=model_b,
            canary_percentage=10,
            promotion_step=20,
        )
        new_pct = manager.promote()
        assert new_pct == 30

    def test_promote_to_100_replaces_primary(self, iris_models) -> None:
        """Full promotion replaces primary with canary."""
        model_a, model_b, _, _ = iris_models
        manager = CanaryManager(
            primary_model=model_a,
            canary_model=model_b,
            canary_percentage=90,
            promotion_step=10,
        )
        manager.promote()
        assert manager.primary_model is model_b
        assert manager.is_active is False
        assert manager.canary_model is None

    def test_rollback_stops_canary(self, iris_models) -> None:
        """Rollback disables canary and routes all traffic to primary."""
        model_a, model_b, _, _ = iris_models
        manager = CanaryManager(
            primary_model=model_a,
            canary_model=model_b,
            canary_percentage=50,
        )
        manager.rollback()
        assert manager.is_active is False
        assert manager.rolled_back is True
        assert manager.canary_percentage == 0

    def test_after_rollback_routes_to_primary(self, iris_models) -> None:
        """After rollback, all requests go to primary."""
        model_a, model_b, _, _ = iris_models
        manager = CanaryManager(
            primary_model=model_a,
            canary_model=model_b,
            canary_percentage=100,
        )
        manager.rollback()

        for _ in range(20):
            model, version = manager.route_request()
            assert model is model_a
            assert version == "primary"

    def test_get_status(self, iris_models) -> None:
        """Status returns complete deployment information."""
        model_a, model_b, _, _ = iris_models
        manager = CanaryManager(
            primary_model=model_a,
            canary_model=model_b,
            canary_percentage=30,
        )
        status = manager.get_status()
        assert status["is_active"] is True
        assert status["canary_percentage"] == 30
        assert "primary_metrics" in status
        assert "canary_metrics" in status
