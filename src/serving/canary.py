"""Canary deployment simulation.

Supports configurable traffic splitting between model versions, per-version
metrics tracking, automated rollback on degradation, and gradual promotion.
"""

import random
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.monitoring.metrics import record_error, record_prediction
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VersionMetrics:
    """Track metrics for a specific model version."""

    total_requests: int = 0
    total_errors: int = 0
    accuracy_sum: float = 0.0
    accuracy_count: int = 0
    latency_sum: float = 0.0

    @property
    def error_rate(self) -> float:
        """Calculate the error rate for this version."""
        if self.total_requests == 0:
            return 0.0
        return self.total_errors / self.total_requests

    @property
    def avg_latency(self) -> float:
        """Calculate the average latency for this version."""
        if self.total_requests == 0:
            return 0.0
        return self.latency_sum / self.total_requests

    @property
    def avg_accuracy(self) -> float:
        """Calculate the average accuracy for this version."""
        if self.accuracy_count == 0:
            return 0.0
        return self.accuracy_sum / self.accuracy_count


class CanaryManager:
    """Manages canary deployments with traffic splitting and rollback.

    Attributes:
        primary_model: The current production model.
        canary_model: The candidate model being tested.
        canary_percentage: Percentage of traffic routed to the canary (0-100).
    """

    def __init__(
        self,
        primary_model: Any = None,
        canary_model: Any = None,
        primary_version: str = "primary",
        canary_version: str = "canary",
        canary_percentage: int = 10,
        rollback_threshold: float = 0.05,
        min_requests_before_eval: int = 50,
        promotion_step: int = 10,
    ) -> None:
        """Initialize the canary manager.

        Args:
            primary_model: The current production model.
            canary_model: The candidate model to test.
            primary_version: Version label for the primary model.
            canary_version: Version label for the canary model.
            canary_percentage: Initial traffic percentage for canary (0-100).
            rollback_threshold: Max acceptable accuracy drop before rollback.
            min_requests_before_eval: Min requests before evaluating canary.
            promotion_step: Percentage increase per promotion step.
        """
        self.primary_model = primary_model
        self.canary_model = canary_model
        self.primary_version = primary_version
        self.canary_version = canary_version
        self.canary_percentage = canary_percentage
        self.rollback_threshold = rollback_threshold
        self.min_requests_before_eval = min_requests_before_eval
        self.promotion_step = promotion_step

        self.primary_metrics = VersionMetrics()
        self.canary_metrics = VersionMetrics()
        self.is_active = canary_model is not None
        self.rolled_back = False

        self._lock = threading.Lock()

    def route_request(self) -> tuple[Any, str]:
        """Route a request to either the primary or canary model.

        Returns:
            Tuple of (model, version_label) based on traffic split.
        """
        if not self.is_active or self.rolled_back:
            return self.primary_model, self.primary_version

        if random.randint(1, 100) <= self.canary_percentage:
            return self.canary_model, self.canary_version

        return self.primary_model, self.primary_version

    def predict(
        self,
        features: np.ndarray,
        model_name: str = "canary-model",
    ) -> tuple[list[int], str, float]:
        """Run prediction through canary routing.

        Args:
            features: Input feature array.
            model_name: Name for metric labeling.

        Returns:
            Tuple of (predictions, version_label, latency_ms).

        Raises:
            RuntimeError: If no model is available.
        """
        model, version = self.route_request()

        if model is None:
            raise RuntimeError("No model available for prediction")

        start = time.time()

        try:
            predictions = model.predict(features)
            if hasattr(predictions, "tolist"):
                pred_list = [int(p) for p in predictions.tolist()]
            else:
                pred_list = [int(p) for p in predictions]
        except Exception:
            latency = time.time() - start
            self._record_request(version, latency, error=True, model_name=model_name)
            raise

        latency = time.time() - start
        self._record_request(version, latency, error=False, model_name=model_name)

        return pred_list, version, latency * 1000

    def _record_request(
        self,
        version: str,
        latency: float,
        error: bool,
        model_name: str,
    ) -> None:
        """Record request metrics for the given version.

        Args:
            version: Model version that handled the request.
            latency: Request latency in seconds.
            error: Whether the request resulted in an error.
            model_name: Name for Prometheus labeling.
        """
        with self._lock:
            metrics = (
                self.canary_metrics
                if version == self.canary_version
                else self.primary_metrics
            )
            metrics.total_requests += 1
            metrics.latency_sum += latency
            if error:
                metrics.total_errors += 1
                record_error(model_name, version, "PredictionError")
            else:
                record_prediction(model_name, version, latency)

    def record_accuracy(self, version: str, accuracy: float) -> None:
        """Record an accuracy measurement for a version.

        Args:
            version: The model version label.
            accuracy: The accuracy value to record.
        """
        with self._lock:
            metrics = (
                self.canary_metrics
                if version == self.canary_version
                else self.primary_metrics
            )
            metrics.accuracy_sum += accuracy
            metrics.accuracy_count += 1

    def evaluate_canary(self) -> dict[str, Any]:
        """Evaluate canary performance and decide on promotion or rollback.

        Returns:
            Dictionary with decision ('promote', 'rollback', or 'continue'),
            metrics comparison, and current canary percentage.
        """
        if not self.is_active or self.rolled_back:
            return {"decision": "inactive", "canary_percentage": 0}

        if self.canary_metrics.total_requests < self.min_requests_before_eval:
            return {
                "decision": "continue",
                "reason": "insufficient requests",
                "canary_requests": self.canary_metrics.total_requests,
                "min_required": self.min_requests_before_eval,
                "canary_percentage": self.canary_percentage,
            }

        primary_acc = self.primary_metrics.avg_accuracy
        canary_acc = self.canary_metrics.avg_accuracy

        accuracy_drop = primary_acc - canary_acc

        canary_error_rate = self.canary_metrics.error_rate
        primary_error_rate = self.primary_metrics.error_rate

        result = {
            "primary_accuracy": primary_acc,
            "canary_accuracy": canary_acc,
            "accuracy_drop": accuracy_drop,
            "primary_error_rate": primary_error_rate,
            "canary_error_rate": canary_error_rate,
            "canary_percentage": self.canary_percentage,
        }

        if accuracy_drop > self.rollback_threshold:
            self.rollback()
            result["decision"] = "rollback"
            result["reason"] = (
                f"accuracy drop {accuracy_drop:.4f} > threshold {self.rollback_threshold}"
            )
        elif canary_error_rate > primary_error_rate * 2 and primary_error_rate > 0:
            self.rollback()
            result["decision"] = "rollback"
            result["reason"] = "canary error rate too high"
        else:
            result["decision"] = "promote"

        return result

    def promote(self) -> int:
        """Increase canary traffic by one promotion step.

        Returns:
            New canary traffic percentage.
        """
        with self._lock:
            self.canary_percentage = min(
                100, self.canary_percentage + self.promotion_step
            )
            logger.info("Canary promoted to %d%% traffic", self.canary_percentage)

            if self.canary_percentage >= 100:
                self.primary_model = self.canary_model
                self.primary_version = self.canary_version
                self.canary_model = None
                self.is_active = False
                logger.info("Canary fully promoted — now primary")

        return self.canary_percentage

    def rollback(self) -> None:
        """Roll back canary deployment and route all traffic to primary."""
        with self._lock:
            self.canary_percentage = 0
            self.is_active = False
            self.rolled_back = True
            self.canary_model = None
            logger.info("Canary rolled back — all traffic to primary")

    def get_status(self) -> dict[str, Any]:
        """Get current canary deployment status.

        Returns:
            Dictionary with deployment state and per-version metrics.
        """
        return {
            "is_active": self.is_active,
            "rolled_back": self.rolled_back,
            "canary_percentage": self.canary_percentage,
            "primary_version": self.primary_version,
            "canary_version": self.canary_version,
            "primary_metrics": {
                "requests": self.primary_metrics.total_requests,
                "errors": self.primary_metrics.total_errors,
                "error_rate": self.primary_metrics.error_rate,
                "avg_latency": self.primary_metrics.avg_latency,
                "avg_accuracy": self.primary_metrics.avg_accuracy,
            },
            "canary_metrics": {
                "requests": self.canary_metrics.total_requests,
                "errors": self.canary_metrics.total_errors,
                "error_rate": self.canary_metrics.error_rate,
                "avg_latency": self.canary_metrics.avg_latency,
                "avg_accuracy": self.canary_metrics.avg_accuracy,
            },
        }
