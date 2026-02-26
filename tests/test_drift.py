"""Tests for accuracy drift detection."""

import pytest

from src.monitoring.drift_detector import DriftDetector, DriftSample


class TestDriftSample:
    """Tests for DriftSample dataclass."""

    def test_create_sample(self) -> None:
        """Creating a sample stores predicted and actual values."""
        s = DriftSample(predicted=1, actual=1)
        assert s.predicted == 1
        assert s.actual == 1
        assert s.timestamp > 0


class TestDriftDetector:
    """Tests for DriftDetector."""

    def test_add_sample(self) -> None:
        """Adding samples increases sample count."""
        dd = DriftDetector(min_samples=5)
        dd.add_sample(1, 1)
        dd.add_sample(0, 0)
        assert dd.sample_count == 2

    def test_add_batch(self) -> None:
        """Adding a batch increases sample count by batch size."""
        dd = DriftDetector(min_samples=5)
        dd.add_batch([1, 0, 1], [1, 0, 0])
        assert dd.sample_count == 3

    def test_add_batch_length_mismatch(self) -> None:
        """Batch with mismatched lengths raises ValueError."""
        dd = DriftDetector()
        with pytest.raises(ValueError, match="Length mismatch"):
            dd.add_batch([1, 2], [1])

    def test_compute_accuracy_insufficient_samples(self) -> None:
        """Returns None when below min_samples threshold."""
        dd = DriftDetector(min_samples=10)
        dd.add_sample(1, 1)
        assert dd.compute_accuracy() is None

    def test_compute_accuracy_correct(self) -> None:
        """Accuracy correctly reflects correct predictions."""
        dd = DriftDetector(min_samples=5, window_size=100)
        for _ in range(8):
            dd.add_sample(1, 1)
        for _ in range(2):
            dd.add_sample(0, 1)

        accuracy = dd.compute_accuracy()
        assert accuracy is not None
        assert abs(accuracy - 0.8) < 1e-6

    def test_compute_accuracy_perfect(self) -> None:
        """Perfect predictions yield 1.0 accuracy."""
        dd = DriftDetector(min_samples=5)
        for i in range(10):
            dd.add_sample(i % 3, i % 3)

        assert dd.compute_accuracy() == 1.0

    def test_sliding_window_limit(self) -> None:
        """Window size limits the number of stored samples."""
        dd = DriftDetector(min_samples=5, window_size=10)
        for i in range(20):
            dd.add_sample(1, 1)
        assert dd.sample_count == 10


class TestDriftDetection:
    """Tests for drift check logic."""

    def test_no_drift_high_accuracy(self) -> None:
        """No drift when accuracy is above threshold."""
        dd = DriftDetector(accuracy_threshold=0.85, min_samples=5)
        for _ in range(10):
            dd.add_sample(1, 1)

        result = dd.check_drift()
        assert result["drift_detected"] is False

    def test_drift_detected_low_accuracy(self) -> None:
        """Drift detected when accuracy drops below threshold."""
        dd = DriftDetector(accuracy_threshold=0.85, min_samples=5)
        for _ in range(5):
            dd.add_sample(1, 1)
        for _ in range(5):
            dd.add_sample(0, 1)

        result = dd.check_drift()
        assert result["drift_detected"] is True
        assert result["current_accuracy"] == 0.5

    def test_check_drift_insufficient_samples(self) -> None:
        """Insufficient samples returns no drift."""
        dd = DriftDetector(min_samples=100)
        dd.add_sample(0, 1)
        result = dd.check_drift()
        assert result["drift_detected"] is False
        assert result["reason"] == "insufficient_samples"

    def test_is_drift_detected_property(self) -> None:
        """Property reflects drift detection state."""
        dd = DriftDetector(accuracy_threshold=0.9, min_samples=5)
        for _ in range(10):
            dd.add_sample(0, 1)

        dd.check_drift()
        assert dd.is_drift_detected is True

    def test_last_accuracy_property(self) -> None:
        """Last accuracy property returns most recent computation."""
        dd = DriftDetector(min_samples=5)
        for _ in range(10):
            dd.add_sample(1, 1)
        dd.compute_accuracy()
        assert dd.last_accuracy == 1.0


class TestDriftReset:
    """Tests for drift detector reset."""

    def test_reset_clears_samples(self) -> None:
        """Reset removes all samples."""
        dd = DriftDetector(min_samples=5)
        for _ in range(10):
            dd.add_sample(1, 1)
        dd.reset()
        assert dd.sample_count == 0

    def test_reset_clears_drift_state(self) -> None:
        """Reset clears drift detection flag."""
        dd = DriftDetector(accuracy_threshold=0.9, min_samples=5)
        for _ in range(10):
            dd.add_sample(0, 1)
        dd.check_drift()
        dd.reset()
        assert dd.is_drift_detected is False
        assert dd.last_accuracy is None
