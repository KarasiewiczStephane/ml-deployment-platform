"""Tests for the ML deployment platform dashboard data generators."""

from src.dashboard.app import (
    generate_canary_data,
    generate_deployment_status,
    generate_health_checks,
    generate_performance_comparison,
)


class TestDeploymentStatus:
    """Tests for the deployment status generator."""

    def test_returns_dict(self) -> None:
        data = generate_deployment_status()
        assert isinstance(data, dict)

    def test_has_staging_and_production(self) -> None:
        data = generate_deployment_status()
        assert "staging" in data
        assert "production" in data

    def test_environment_keys(self) -> None:
        data = generate_deployment_status()
        required_keys = {
            "model_version",
            "status",
            "deployed_at",
            "replicas",
            "avg_latency_ms",
            "requests_per_min",
            "error_rate",
            "uptime_hours",
        }
        for env in ("staging", "production"):
            assert required_keys.issubset(data[env].keys())

    def test_replicas_structure(self) -> None:
        data = generate_deployment_status()
        for env in ("staging", "production"):
            replicas = data[env]["replicas"]
            assert "ready" in replicas
            assert "desired" in replicas
            assert replicas["ready"] <= replicas["desired"]


class TestCanaryData:
    """Tests for the canary deployment data generator."""

    def test_returns_dict(self) -> None:
        data = generate_canary_data()
        assert isinstance(data, dict)

    def test_has_required_keys(self) -> None:
        data = generate_canary_data()
        required = {
            "is_active",
            "primary_version",
            "canary_version",
            "canary_percentage",
            "primary_metrics",
            "canary_metrics",
            "traffic_history",
        }
        assert required.issubset(data.keys())

    def test_canary_percentage_range(self) -> None:
        data = generate_canary_data()
        assert 0 <= data["canary_percentage"] <= 100

    def test_traffic_history_not_empty(self) -> None:
        data = generate_canary_data()
        assert len(data["traffic_history"]) > 0

    def test_metrics_keys(self) -> None:
        data = generate_canary_data()
        metric_keys = {
            "requests",
            "accuracy",
            "avg_latency_ms",
            "error_rate",
            "p99_latency_ms",
        }
        assert metric_keys.issubset(data["primary_metrics"].keys())
        assert metric_keys.issubset(data["canary_metrics"].keys())


class TestPerformanceComparison:
    """Tests for the performance comparison generator."""

    def test_returns_list(self) -> None:
        data = generate_performance_comparison()
        assert isinstance(data, list)

    def test_has_entries(self) -> None:
        data = generate_performance_comparison()
        assert len(data) > 0

    def test_entry_keys(self) -> None:
        data = generate_performance_comparison()
        for entry in data:
            assert "metric" in entry
            assert "v2.3.1 (Primary)" in entry
            assert "v2.4.0-rc1 (Canary)" in entry


class TestHealthChecks:
    """Tests for the health checks generator."""

    def test_returns_list(self) -> None:
        data = generate_health_checks()
        assert isinstance(data, list)

    def test_has_entries(self) -> None:
        data = generate_health_checks()
        assert len(data) > 0

    def test_entry_keys(self) -> None:
        data = generate_health_checks()
        required_keys = {"service", "status", "latency_ms", "last_check"}
        for entry in data:
            assert required_keys.issubset(entry.keys())

    def test_valid_statuses(self) -> None:
        data = generate_health_checks()
        valid = {"healthy", "degraded", "unhealthy"}
        for entry in data:
            assert entry["status"] in valid
