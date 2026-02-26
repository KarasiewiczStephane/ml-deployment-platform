"""Automated retraining trigger.

Monitors model accuracy via drift detection and triggers retraining
when performance drops below threshold. Registers new models in MLflow.
"""

import time
from typing import Any

from src.monitoring.drift_detector import DriftDetector
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RetrainingTrigger:
    """Automated retraining manager.

    Watches a DriftDetector for accuracy drops and triggers model
    retraining when the degradation exceeds the configured threshold.

    Attributes:
        drift_detector: The drift detector to monitor.
        accuracy_drop_threshold: Max acceptable accuracy drop.
        min_samples_for_retrain: Min samples before retraining eligible.
    """

    def __init__(
        self,
        drift_detector: DriftDetector,
        accuracy_drop_threshold: float = 0.05,
        min_samples_for_retrain: int = 200,
        max_retrain_frequency_seconds: int = 3600,
        model_name: str = "model",
        tracking_uri: str | None = None,
        config_path: str | None = None,
    ) -> None:
        """Initialize the retraining trigger.

        Args:
            drift_detector: DriftDetector instance to monitor.
            accuracy_drop_threshold: Max accuracy drop to tolerate.
            min_samples_for_retrain: Min samples before retraining.
            max_retrain_frequency_seconds: Cooldown between retraining runs.
            model_name: Model name for MLflow registration.
            tracking_uri: MLflow tracking URI override.
            config_path: Config file path override.
        """
        self.drift_detector = drift_detector
        self.accuracy_drop_threshold = accuracy_drop_threshold
        self.min_samples_for_retrain = min_samples_for_retrain
        self.max_retrain_frequency_seconds = max_retrain_frequency_seconds
        self.model_name = model_name
        self.tracking_uri = tracking_uri
        self.config_path = config_path

        self._last_retrain_time: float = 0
        self._retrain_count: int = 0
        self._last_result: dict[str, Any] | None = None

    def check_and_retrain(self) -> dict[str, Any]:
        """Check drift status and trigger retraining if needed.

        Returns:
            Dictionary with action taken, drift status, and retraining result.
        """
        drift_status = self.drift_detector.check_drift()

        if not drift_status["drift_detected"]:
            return {
                "action": "none",
                "reason": "no drift detected",
                "drift_status": drift_status,
            }

        if self.drift_detector.sample_count < self.min_samples_for_retrain:
            return {
                "action": "none",
                "reason": "insufficient samples for retraining",
                "sample_count": self.drift_detector.sample_count,
                "min_required": self.min_samples_for_retrain,
                "drift_status": drift_status,
            }

        now = time.time()
        elapsed = now - self._last_retrain_time
        if self._last_retrain_time > 0 and elapsed < self.max_retrain_frequency_seconds:
            return {
                "action": "none",
                "reason": "cooldown period active",
                "seconds_remaining": int(self.max_retrain_frequency_seconds - elapsed),
                "drift_status": drift_status,
            }

        logger.info(
            "Triggering retraining — accuracy=%.4f, threshold=%.4f",
            drift_status.get("current_accuracy", 0),
            self.drift_detector.accuracy_threshold,
        )

        result = self._execute_retrain()
        return result

    def _execute_retrain(self) -> dict[str, Any]:
        """Execute the retraining pipeline and register the new model.

        Returns:
            Dictionary with retrain outcome including run_id and metrics.
        """
        try:
            from src.training.train import train_model

            train_result = train_model(
                model_type="random_forest",
                config_path=self.config_path,
                tracking_uri=self.tracking_uri,
            )

            try:
                from src.serving.model_loader import register_model

                mv = register_model(
                    run_id=train_result["run_id"],
                    model_name=self.model_name,
                    tracking_uri=self.tracking_uri,
                    tags={"retrained": "true", "trigger": "drift_detection"},
                )
                registered_version = mv.version
            except Exception as e:
                logger.warning("Model registration failed: %s", e)
                registered_version = None

            self._last_retrain_time = time.time()
            self._retrain_count += 1

            self.drift_detector.reset()

            result = {
                "action": "retrained",
                "run_id": train_result["run_id"],
                "metrics": train_result["metrics"],
                "registered_version": registered_version,
                "retrain_count": self._retrain_count,
            }
            self._last_result = result

            logger.info(
                "Retraining complete — run_id=%s, accuracy=%.4f",
                train_result["run_id"],
                train_result["metrics"]["accuracy"],
            )

            return result

        except Exception as e:
            logger.error("Retraining failed: %s", e)
            return {
                "action": "failed",
                "error": str(e),
            }

    @property
    def retrain_count(self) -> int:
        """Return the total number of retraining runs completed."""
        return self._retrain_count

    @property
    def last_result(self) -> dict[str, Any] | None:
        """Return the result of the last retraining run."""
        return self._last_result

    def get_status(self) -> dict[str, Any]:
        """Get current retraining trigger status.

        Returns:
            Dictionary with trigger configuration and state.
        """
        return {
            "retrain_count": self._retrain_count,
            "last_retrain_time": self._last_retrain_time,
            "accuracy_drop_threshold": self.accuracy_drop_threshold,
            "min_samples_for_retrain": self.min_samples_for_retrain,
            "cooldown_seconds": self.max_retrain_frequency_seconds,
            "drift_detected": self.drift_detector.is_drift_detected,
            "sample_count": self.drift_detector.sample_count,
            "last_accuracy": self.drift_detector.last_accuracy,
        }
