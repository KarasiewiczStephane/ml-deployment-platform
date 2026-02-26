"""Tests for automated retraining trigger."""

import tempfile
from pathlib import Path


from src.monitoring.drift_detector import DriftDetector
from src.retraining.trigger import RetrainingTrigger


class TestRetrainingTrigger:
    """Tests for RetrainingTrigger."""

    def _make_trigger(
        self,
        config_file: Path,
        accuracy_threshold: float = 0.85,
        min_samples: int = 10,
        min_retrain_samples: int = 10,
    ) -> RetrainingTrigger:
        """Create a trigger with a configured drift detector."""
        dd = DriftDetector(
            accuracy_threshold=accuracy_threshold,
            min_samples=min_samples,
            model_name="retrain-test-model",
            model_version="1",
        )
        return RetrainingTrigger(
            drift_detector=dd,
            accuracy_drop_threshold=0.05,
            min_samples_for_retrain=min_retrain_samples,
            max_retrain_frequency_seconds=0,
            model_name="retrain-test-model",
            config_path=str(config_file),
        )

    def test_no_action_without_drift(self, config_file: Path) -> None:
        """No retraining when there's no drift."""
        trigger = self._make_trigger(config_file)
        for _ in range(20):
            trigger.drift_detector.add_sample(1, 1)

        result = trigger.check_and_retrain()
        assert result["action"] == "none"
        assert result["reason"] == "no drift detected"

    def test_no_action_insufficient_samples(self, config_file: Path) -> None:
        """No retraining when drift detected but samples insufficient."""
        trigger = self._make_trigger(
            config_file, min_samples=5, min_retrain_samples=100
        )
        for _ in range(5):
            trigger.drift_detector.add_sample(0, 1)

        result = trigger.check_and_retrain()
        assert result["action"] == "none"
        assert "insufficient" in result["reason"]

    def test_retrain_on_drift(self, config_file: Path) -> None:
        """Retraining triggers when drift is detected with enough samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking_uri = f"sqlite:///{tmpdir}/mlflow.db"

            dd = DriftDetector(
                accuracy_threshold=0.95,
                min_samples=5,
                model_name="retrain-test-model",
                model_version="1",
            )
            trigger = RetrainingTrigger(
                drift_detector=dd,
                min_samples_for_retrain=5,
                max_retrain_frequency_seconds=0,
                model_name="retrain-test-model",
                tracking_uri=tracking_uri,
                config_path=str(config_file),
            )

            for _ in range(5):
                dd.add_sample(0, 1)
            for _ in range(5):
                dd.add_sample(1, 1)

            result = trigger.check_and_retrain()
            assert result["action"] == "retrained"
            assert "run_id" in result
            assert result["metrics"]["accuracy"] > 0

    def test_cooldown_prevents_retrain(self, config_file: Path) -> None:
        """Cooldown period prevents rapid retraining."""
        dd = DriftDetector(
            accuracy_threshold=0.95,
            min_samples=5,
            model_name="cooldown-test",
            model_version="1",
        )
        trigger = RetrainingTrigger(
            drift_detector=dd,
            min_samples_for_retrain=5,
            max_retrain_frequency_seconds=9999,
            model_name="cooldown-test",
            config_path=str(config_file),
        )

        trigger._last_retrain_time = 9999999999

        for _ in range(10):
            dd.add_sample(0, 1)

        result = trigger.check_and_retrain()
        assert result["action"] == "none"
        assert "cooldown" in result["reason"]

    def test_retrain_count_increments(self, config_file: Path) -> None:
        """Retrain count increases after each successful retrain."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking_uri = f"sqlite:///{tmpdir}/mlflow.db"

            dd = DriftDetector(
                accuracy_threshold=0.95,
                min_samples=5,
                model_name="count-test",
                model_version="1",
            )
            trigger = RetrainingTrigger(
                drift_detector=dd,
                min_samples_for_retrain=5,
                max_retrain_frequency_seconds=0,
                model_name="count-test",
                tracking_uri=tracking_uri,
                config_path=str(config_file),
            )

            for _ in range(10):
                dd.add_sample(0, 1)

            trigger.check_and_retrain()
            assert trigger.retrain_count == 1

    def test_get_status(self, config_file: Path) -> None:
        """Status returns all trigger information."""
        trigger = self._make_trigger(config_file)
        status = trigger.get_status()
        assert "retrain_count" in status
        assert "drift_detected" in status
        assert "sample_count" in status

    def test_last_result_initially_none(self, config_file: Path) -> None:
        """Last result is None before any retraining."""
        trigger = self._make_trigger(config_file)
        assert trigger.last_result is None
