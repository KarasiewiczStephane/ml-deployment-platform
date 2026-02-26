"""Accuracy drift detection for deployed models.

Monitors model accuracy on recent labeled data using a sliding window
and flags when accuracy drops below a configurable threshold.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field

from src.monitoring.metrics import update_accuracy
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DriftSample:
    """A single labeled sample for drift detection."""

    predicted: int
    actual: int
    timestamp: float = field(default_factory=time.time)


class DriftDetector:
    """Monitors model accuracy and detects performance drift.

    Maintains a sliding window of labeled predictions and triggers
    alerts when accuracy drops below the configured threshold.

    Attributes:
        accuracy_threshold: Minimum acceptable accuracy.
        window_size: Number of recent samples to evaluate.
        min_samples: Minimum samples required before detection starts.
    """

    def __init__(
        self,
        accuracy_threshold: float = 0.85,
        window_size: int = 500,
        min_samples: int = 100,
        model_name: str = "model",
        model_version: str = "1",
    ) -> None:
        """Initialize the drift detector.

        Args:
            accuracy_threshold: Minimum acceptable accuracy before triggering.
            window_size: Size of the sliding window.
            min_samples: Min samples before drift detection activates.
            model_name: Model name for metric labels.
            model_version: Model version for metric labels.
        """
        self.accuracy_threshold = accuracy_threshold
        self.window_size = window_size
        self.min_samples = min_samples
        self.model_name = model_name
        self.model_version = model_version

        self._samples: deque[DriftSample] = deque(maxlen=window_size)
        self._lock = threading.Lock()
        self._drift_detected = False
        self._last_accuracy: float | None = None

    def add_sample(self, predicted: int, actual: int) -> None:
        """Add a labeled prediction sample for drift monitoring.

        Args:
            predicted: The model's predicted label.
            actual: The true label.
        """
        with self._lock:
            self._samples.append(DriftSample(predicted=predicted, actual=actual))

    def add_batch(self, predictions: list[int], actuals: list[int]) -> None:
        """Add a batch of labeled predictions.

        Args:
            predictions: List of predicted labels.
            actuals: List of true labels.

        Raises:
            ValueError: If predictions and actuals have different lengths.
        """
        if len(predictions) != len(actuals):
            raise ValueError(
                f"Length mismatch: {len(predictions)} predictions vs {len(actuals)} actuals"
            )

        with self._lock:
            for pred, actual in zip(predictions, actuals):
                self._samples.append(DriftSample(predicted=pred, actual=actual))

    def compute_accuracy(self) -> float | None:
        """Compute accuracy on the current sliding window.

        Returns:
            Current accuracy as a float, or None if insufficient samples.
        """
        with self._lock:
            if len(self._samples) < self.min_samples:
                return None

            correct = sum(1 for s in self._samples if s.predicted == s.actual)
            accuracy = correct / len(self._samples)

        self._last_accuracy = accuracy

        update_accuracy(self.model_name, self.model_version, accuracy)

        return accuracy

    def check_drift(self) -> dict:
        """Check if model accuracy has drifted below threshold.

        Returns:
            Dictionary with drift status, current accuracy, threshold,
            sample count, and whether drift was detected.
        """
        accuracy = self.compute_accuracy()

        if accuracy is None:
            return {
                "drift_detected": False,
                "reason": "insufficient_samples",
                "sample_count": len(self._samples),
                "min_samples": self.min_samples,
            }

        drift_detected = accuracy < self.accuracy_threshold
        self._drift_detected = drift_detected

        result = {
            "drift_detected": drift_detected,
            "current_accuracy": accuracy,
            "accuracy_threshold": self.accuracy_threshold,
            "sample_count": len(self._samples),
            "model_name": self.model_name,
            "model_version": self.model_version,
        }

        if drift_detected:
            logger.warning(
                "Drift detected — accuracy=%.4f < threshold=%.4f",
                accuracy,
                self.accuracy_threshold,
            )

        return result

    def reset(self) -> None:
        """Clear all samples and reset drift state."""
        with self._lock:
            self._samples.clear()
            self._drift_detected = False
            self._last_accuracy = None

    @property
    def sample_count(self) -> int:
        """Return the current number of samples in the window."""
        return len(self._samples)

    @property
    def is_drift_detected(self) -> bool:
        """Return whether drift is currently flagged."""
        return self._drift_detected

    @property
    def last_accuracy(self) -> float | None:
        """Return the last computed accuracy value."""
        return self._last_accuracy
