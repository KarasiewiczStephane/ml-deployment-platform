"""Prometheus metric definitions and collectors.

Defines prediction latency histogram, prediction counter, model accuracy
gauge, and error rate counter for monitoring the serving layer.
"""

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)

REGISTRY = CollectorRegistry()

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time spent processing prediction requests",
    ["model_name", "model_version"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    registry=REGISTRY,
)

PREDICTION_COUNT = Counter(
    "prediction_count_total",
    "Total number of predictions made",
    ["model_name", "model_version"],
    registry=REGISTRY,
)

PREDICTION_ERROR_COUNT = Counter(
    "prediction_error_total",
    "Total number of prediction errors",
    ["model_name", "model_version", "error_type"],
    registry=REGISTRY,
)

MODEL_ACCURACY = Gauge(
    "model_accuracy",
    "Current model accuracy on recent data",
    ["model_name", "model_version"],
    registry=REGISTRY,
)


def record_prediction(
    model_name: str,
    model_version: str,
    latency_seconds: float,
) -> None:
    """Record a successful prediction with latency.

    Args:
        model_name: Name of the model that made the prediction.
        model_version: Version of the model.
        latency_seconds: Time taken for the prediction in seconds.
    """
    PREDICTION_LATENCY.labels(
        model_name=model_name, model_version=model_version
    ).observe(latency_seconds)
    PREDICTION_COUNT.labels(model_name=model_name, model_version=model_version).inc()


def record_error(
    model_name: str,
    model_version: str,
    error_type: str,
) -> None:
    """Record a prediction error.

    Args:
        model_name: Name of the model.
        model_version: Version of the model.
        error_type: Type/class of the error encountered.
    """
    PREDICTION_ERROR_COUNT.labels(
        model_name=model_name,
        model_version=model_version,
        error_type=error_type,
    ).inc()


def update_accuracy(
    model_name: str,
    model_version: str,
    accuracy: float,
) -> None:
    """Update the model accuracy gauge.

    Args:
        model_name: Name of the model.
        model_version: Version of the model.
        accuracy: Current accuracy value (0.0 to 1.0).
    """
    MODEL_ACCURACY.labels(model_name=model_name, model_version=model_version).set(
        accuracy
    )


def get_metrics() -> bytes:
    """Generate Prometheus-formatted metrics output.

    Returns:
        Bytes containing all registered metrics in Prometheus exposition format.
    """
    return generate_latest(REGISTRY)
