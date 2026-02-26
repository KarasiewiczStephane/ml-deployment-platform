"""Tests for the FastAPI model serving application."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

from src.serving.app import (
    HealthResponse,
    ModelState,
    PredictRequest,
    PredictResponse,
    app,
    model_state,
)


@pytest.fixture
def trained_model():
    """Return a small trained RandomForest model."""
    from src.training.train import load_dataset

    X_train, X_test, y_train, y_test, _ = load_dataset("iris")
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X_train, y_train)
    return model


@pytest.fixture
def client_with_model(trained_model):
    """Create a test client with a loaded model."""
    original_model = model_state.model
    original_loaded = model_state.loaded
    original_name = model_state.model_name
    original_version = model_state.model_version

    model_state.model = trained_model
    model_state.model_name = "test-model"
    model_state.model_version = "1"
    model_state.loaded = True

    with patch("src.serving.app._try_load_model"):
        client = TestClient(app)
        yield client

    model_state.model = original_model
    model_state.loaded = original_loaded
    model_state.model_name = original_name
    model_state.model_version = original_version


@pytest.fixture
def client_no_model():
    """Create a test client without a loaded model."""
    original_model = model_state.model
    original_loaded = model_state.loaded
    original_name = model_state.model_name
    original_version = model_state.model_version

    model_state.model = None
    model_state.model_name = ""
    model_state.model_version = ""
    model_state.loaded = False

    with patch("src.serving.app._try_load_model"):
        client = TestClient(app)
        yield client

    model_state.model = original_model
    model_state.loaded = original_loaded
    model_state.model_name = original_name
    model_state.model_version = original_version


class TestPredictEndpoint:
    """Tests for POST /predict."""

    def test_predict_success(self, client_with_model) -> None:
        """Prediction returns valid results for iris features."""
        response = client_with_model.post(
            "/predict",
            json={"features": [[5.1, 3.5, 1.4, 0.2]]},
        )
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 1
        assert isinstance(data["predictions"][0], int)
        assert data["model_name"] == "test-model"
        assert data["latency_ms"] >= 0

    def test_predict_multiple_samples(self, client_with_model) -> None:
        """Prediction handles multiple samples."""
        features = [[5.1, 3.5, 1.4, 0.2], [6.7, 3.1, 4.7, 1.5]]
        response = client_with_model.post(
            "/predict",
            json={"features": features},
        )
        assert response.status_code == 200
        assert len(response.json()["predictions"]) == 2

    def test_predict_no_model_returns_503(self, client_no_model) -> None:
        """Prediction without a loaded model returns 503."""
        response = client_no_model.post(
            "/predict",
            json={"features": [[5.1, 3.5, 1.4, 0.2]]},
        )
        assert response.status_code == 503

    def test_predict_empty_features_returns_422(self, client_with_model) -> None:
        """Empty features list returns 422 validation error."""
        response = client_with_model.post(
            "/predict",
            json={"features": []},
        )
        assert response.status_code == 422

    def test_predict_invalid_json(self, client_with_model) -> None:
        """Invalid JSON body returns 422."""
        response = client_with_model.post(
            "/predict",
            json={"wrong_field": [[1, 2, 3]]},
        )
        assert response.status_code == 422


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_with_model(self, client_with_model) -> None:
        """Health check with loaded model returns healthy status."""
        response = client_with_model.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["model_name"] == "test-model"

    def test_health_without_model(self, client_no_model) -> None:
        """Health check without model returns degraded status."""
        response = client_no_model.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False


class TestReloadEndpoint:
    """Tests for POST /reload."""

    def test_reload_success(self, client_with_model) -> None:
        """Reload endpoint responds successfully."""
        response = client_with_model.post("/reload")
        assert response.status_code == 200


class TestModelState:
    """Tests for ModelState class."""

    def test_initial_state(self) -> None:
        """New ModelState starts unloaded."""
        state = ModelState()
        assert state.loaded is False
        assert state.model is None

    def test_update_state(self) -> None:
        """Updating state sets all fields correctly."""
        state = ModelState()
        mock_model = MagicMock()
        state.update(mock_model, "my-model", "3")
        assert state.loaded is True
        assert state.model is mock_model
        assert state.model_name == "my-model"
        assert state.model_version == "3"


class TestPydanticModels:
    """Tests for request/response Pydantic models."""

    def test_predict_request_valid(self) -> None:
        """Valid predict request parses correctly."""
        req = PredictRequest(features=[[1.0, 2.0, 3.0, 4.0]])
        assert len(req.features) == 1

    def test_predict_response_valid(self) -> None:
        """Valid predict response serializes correctly."""
        resp = PredictResponse(
            predictions=[0, 1],
            model_name="test",
            model_version="1",
            latency_ms=1.5,
        )
        assert resp.predictions == [0, 1]

    def test_health_response_valid(self) -> None:
        """Valid health response serializes correctly."""
        resp = HealthResponse(
            status="healthy",
            model_name="test",
            model_version="1",
            model_loaded=True,
        )
        assert resp.status == "healthy"
