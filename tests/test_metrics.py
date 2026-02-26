"""Tests for Prometheus metric collectors."""

from unittest.mock import patch


from src.monitoring.metrics import (
    MODEL_ACCURACY,
    PREDICTION_COUNT,
    PREDICTION_ERROR_COUNT,
    get_metrics,
    record_error,
    record_prediction,
    update_accuracy,
)


class TestRecordPrediction:
    """Tests for prediction recording."""

    def test_record_prediction_increments_counter(self) -> None:
        """Recording a prediction increments the count."""
        before = PREDICTION_COUNT.labels(
            model_name="test-model", model_version="1"
        )._value.get()
        record_prediction("test-model", "1", 0.05)
        after = PREDICTION_COUNT.labels(
            model_name="test-model", model_version="1"
        )._value.get()
        assert after == before + 1

    def test_record_prediction_observes_latency(self) -> None:
        """Recording a prediction observes the latency histogram."""
        record_prediction("test-model", "1", 0.123)
        metrics_output = get_metrics().decode()
        assert "prediction_latency_seconds" in metrics_output

    def test_record_multiple_predictions(self) -> None:
        """Multiple predictions accumulate correctly."""
        initial = PREDICTION_COUNT.labels(
            model_name="multi-model", model_version="2"
        )._value.get()
        for _ in range(5):
            record_prediction("multi-model", "2", 0.01)
        final = PREDICTION_COUNT.labels(
            model_name="multi-model", model_version="2"
        )._value.get()
        assert final == initial + 5


class TestRecordError:
    """Tests for error recording."""

    def test_record_error_increments(self) -> None:
        """Recording an error increments the error counter."""
        before = PREDICTION_ERROR_COUNT.labels(
            model_name="test-model", model_version="1", error_type="ValueError"
        )._value.get()
        record_error("test-model", "1", "ValueError")
        after = PREDICTION_ERROR_COUNT.labels(
            model_name="test-model", model_version="1", error_type="ValueError"
        )._value.get()
        assert after == before + 1

    def test_record_different_error_types(self) -> None:
        """Different error types are tracked separately."""
        record_error("err-model", "1", "TypeError")
        record_error("err-model", "1", "RuntimeError")
        type_count = PREDICTION_ERROR_COUNT.labels(
            model_name="err-model", model_version="1", error_type="TypeError"
        )._value.get()
        runtime_count = PREDICTION_ERROR_COUNT.labels(
            model_name="err-model", model_version="1", error_type="RuntimeError"
        )._value.get()
        assert type_count >= 1
        assert runtime_count >= 1


class TestUpdateAccuracy:
    """Tests for accuracy gauge updates."""

    def test_update_accuracy_sets_value(self) -> None:
        """Updating accuracy sets the gauge to the given value."""
        update_accuracy("acc-model", "1", 0.95)
        value = MODEL_ACCURACY.labels(
            model_name="acc-model", model_version="1"
        )._value.get()
        assert abs(value - 0.95) < 1e-6

    def test_update_accuracy_overwrites(self) -> None:
        """Updating accuracy replaces the previous value."""
        update_accuracy("acc-model-2", "1", 0.90)
        update_accuracy("acc-model-2", "1", 0.85)
        value = MODEL_ACCURACY.labels(
            model_name="acc-model-2", model_version="1"
        )._value.get()
        assert abs(value - 0.85) < 1e-6


class TestGetMetrics:
    """Tests for metrics exposition."""

    def test_returns_bytes(self) -> None:
        """get_metrics returns bytes output."""
        result = get_metrics()
        assert isinstance(result, bytes)

    def test_contains_metric_names(self) -> None:
        """Output contains expected metric family names."""
        record_prediction("output-test", "1", 0.01)
        output = get_metrics().decode()
        assert "prediction_latency_seconds" in output
        assert "prediction_count_total" in output

    def test_prometheus_format(self) -> None:
        """Output follows Prometheus text format with # HELP and # TYPE."""
        output = get_metrics().decode()
        assert "# HELP" in output
        assert "# TYPE" in output


class TestMetricsEndpoint:
    """Tests for the /metrics FastAPI endpoint."""

    def test_metrics_endpoint_returns_200(self) -> None:
        """The /metrics endpoint returns a 200 response."""

        from fastapi.testclient import TestClient

        from src.serving.app import app

        with patch("src.serving.app._try_load_model"):
            client = TestClient(app)
            response = client.get("/metrics")
            assert response.status_code == 200
            assert "prediction_latency_seconds" in response.text
