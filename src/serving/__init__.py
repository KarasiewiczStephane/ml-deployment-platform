"""Model serving with FastAPI and canary deployment."""

from src.serving.model_loader import (
    compare_model_versions,
    get_model_info,
    load_model_by_stage,
    register_model,
    transition_model_stage,
)

__all__ = [
    "compare_model_versions",
    "get_model_info",
    "load_model_by_stage",
    "register_model",
    "transition_model_stage",
]
