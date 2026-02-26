"""FastAPI model serving application.

Provides REST endpoints for model inference, health checks, and metrics.
Supports automatic model loading from MLflow registry and hot-reload.
"""

import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PredictRequest(BaseModel):
    """Request schema for prediction endpoint."""

    features: list[list[float]] = Field(
        ...,
        description="2D array of feature values. Each inner list is one sample.",
        min_length=1,
    )


class PredictResponse(BaseModel):
    """Response schema for prediction endpoint."""

    predictions: list[int]
    model_name: str
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""

    status: str
    model_name: str
    model_version: str
    model_loaded: bool


class ModelState:
    """Holds the currently loaded model and its metadata."""

    def __init__(self) -> None:
        self.model: Any = None
        self.model_name: str = ""
        self.model_version: str = ""
        self.loaded: bool = False

    def update(self, model: Any, name: str, version: str) -> None:
        """Update the loaded model state.

        Args:
            model: The loaded model object.
            name: Model name from registry.
            version: Model version string.
        """
        self.model = model
        self.model_name = name
        self.model_version = version
        self.loaded = True
        logger.info("Model updated: %s v%s", name, version)


model_state = ModelState()


def _try_load_model(state: ModelState) -> None:
    """Attempt to load the model from MLflow registry.

    Args:
        state: ModelState instance to update.
    """
    try:
        config = load_config()
        serving_config = config["serving"]
        model_name = serving_config["model_name"]
        model_stage = serving_config.get("model_stage", "Production").lower()

        from src.serving.model_loader import load_model_by_stage

        model = load_model_by_stage(
            model_name=model_name,
            stage=model_stage,
            tracking_uri=config["mlflow"]["tracking_uri"],
        )
        state.update(model, model_name, model_stage)
    except Exception as e:
        logger.warning("Could not load model from registry: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown.

    Args:
        app: The FastAPI application instance.
    """
    logger.info("Starting model serving application")
    _try_load_model(model_state)
    yield
    logger.info("Shutting down model serving application")


app = FastAPI(
    title="ML Deployment Platform API",
    description="Model serving with MLflow integration",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Run inference on the loaded model.

    Args:
        request: Prediction request with feature values.

    Returns:
        PredictResponse with predictions, model info, and latency.

    Raises:
        HTTPException: If no model is loaded or prediction fails.
    """
    if not model_state.loaded or model_state.model is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    start_time = time.time()

    try:
        features = np.array(request.features)
        predictions = model_state.model.predict(features)

        if hasattr(predictions, "tolist"):
            pred_list = predictions.tolist()
        else:
            pred_list = list(predictions)

        pred_list = [int(p) for p in pred_list]

    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    latency_ms = (time.time() - start_time) * 1000

    config = load_config()
    if config["serving"].get("log_predictions", False):
        logger.info(
            "Prediction — samples=%d, latency=%.2fms",
            len(pred_list),
            latency_ms,
        )

    return PredictResponse(
        predictions=pred_list,
        model_name=model_state.model_name,
        model_version=model_state.model_version,
        latency_ms=round(latency_ms, 2),
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Check API health and model status.

    Returns:
        HealthResponse with model loading status.
    """
    return HealthResponse(
        status="healthy" if model_state.loaded else "degraded",
        model_name=model_state.model_name,
        model_version=model_state.model_version,
        model_loaded=model_state.loaded,
    )


@app.post("/reload")
async def reload_model() -> dict[str, str]:
    """Trigger a model reload from the MLflow registry.

    Returns:
        Status message indicating reload result.
    """
    _try_load_model(model_state)

    if model_state.loaded:
        return {
            "status": "reloaded",
            "model_name": model_state.model_name,
            "model_version": model_state.model_version,
        }

    return {"status": "failed", "detail": "Could not load model"}
