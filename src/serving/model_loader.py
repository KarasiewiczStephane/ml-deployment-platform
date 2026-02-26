"""MLflow Model Registry integration.

Handles model registration, stage transitions, loading by stage,
and model version comparison.
"""

from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_mlflow_client(tracking_uri: str | None = None) -> MlflowClient:
    """Create an MLflow client instance.

    Args:
        tracking_uri: Optional tracking URI override.

    Returns:
        Configured MlflowClient.
    """
    if tracking_uri is None:
        config = load_config()
        tracking_uri = config["mlflow"]["tracking_uri"]

    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)


def register_model(
    run_id: str,
    model_name: str,
    tracking_uri: str | None = None,
    description: str | None = None,
    tags: dict[str, str] | None = None,
) -> Any:
    """Register a model from an MLflow run in the Model Registry.

    Args:
        run_id: MLflow run ID containing the logged model.
        model_name: Name for the registered model.
        tracking_uri: Optional tracking URI override.
        description: Optional model description.
        tags: Optional key-value tags for the model version.

    Returns:
        The registered ModelVersion object.
    """
    client = get_mlflow_client(tracking_uri)
    model_uri = f"runs:/{run_id}/model"

    result = mlflow.register_model(model_uri, model_name)

    if description:
        client.update_model_version(
            name=model_name,
            version=result.version,
            description=description,
        )

    if tags:
        for key, value in tags.items():
            client.set_model_version_tag(
                name=model_name,
                version=result.version,
                key=key,
                value=value,
            )

    logger.info(
        "Registered model '%s' version %s from run %s",
        model_name,
        result.version,
        run_id,
    )

    return result


def transition_model_stage(
    model_name: str,
    version: str,
    stage: str,
    tracking_uri: str | None = None,
) -> Any:
    """Transition a model version to a new stage using aliases.

    Args:
        model_name: Name of the registered model.
        version: Version number to transition.
        stage: Target stage ('staging', 'production', 'archived').
        tracking_uri: Optional tracking URI override.

    Returns:
        The updated ModelVersion object.
    """
    client = get_mlflow_client(tracking_uri)

    stage_lower = stage.lower()
    valid_stages = {"staging", "production", "archived"}
    if stage_lower not in valid_stages:
        raise ValueError(f"Invalid stage '{stage}'. Must be one of {valid_stages}")

    alias_name = stage_lower

    if stage_lower != "archived":
        client.set_registered_model_alias(
            name=model_name,
            alias=alias_name,
            version=version,
        )
        logger.info(
            "Set alias '%s' on model '%s' version %s",
            alias_name,
            model_name,
            version,
        )
    else:
        for alias in ("staging", "production"):
            try:
                alias_version = client.get_model_version_by_alias(model_name, alias)
                if str(alias_version.version) == str(version):
                    client.delete_registered_model_alias(name=model_name, alias=alias)
            except Exception:
                pass

        logger.info(
            "Archived model '%s' version %s (removed active aliases)",
            model_name,
            version,
        )

    return client.get_model_version(name=model_name, version=version)


def load_model_by_stage(
    model_name: str,
    stage: str = "production",
    tracking_uri: str | None = None,
) -> Any:
    """Load a model from the registry by its stage alias.

    Args:
        model_name: Name of the registered model.
        stage: Stage alias to load ('staging' or 'production').
        tracking_uri: Optional tracking URI override.

    Returns:
        The loaded model object.

    Raises:
        ValueError: If no model is found at the given stage.
    """
    client = get_mlflow_client(tracking_uri)

    try:
        model_version = client.get_model_version_by_alias(model_name, stage.lower())
    except Exception as e:
        raise ValueError(
            f"No model found with alias '{stage}' for '{model_name}'"
        ) from e

    model_uri = f"models:/{model_name}@{stage.lower()}"
    model = mlflow.pyfunc.load_model(model_uri)

    logger.info(
        "Loaded model '%s' version %s (stage=%s)",
        model_name,
        model_version.version,
        stage,
    )

    return model


def get_model_info(
    model_name: str,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Get information about all versions of a registered model.

    Args:
        model_name: Name of the registered model.
        tracking_uri: Optional tracking URI override.

    Returns:
        Dictionary with model name, versions list, and alias mappings.
    """
    client = get_mlflow_client(tracking_uri)

    try:
        registered_model = client.get_registered_model(model_name)
    except Exception as e:
        raise ValueError(f"Model '{model_name}' not found in registry") from e

    versions = []
    for mv in client.search_model_versions(f"name='{model_name}'"):
        versions.append(
            {
                "version": mv.version,
                "status": mv.status,
                "description": mv.description or "",
                "tags": dict(mv.tags) if mv.tags else {},
                "run_id": mv.run_id,
            }
        )

    aliases = {}
    if hasattr(registered_model, "aliases") and registered_model.aliases:
        aliases = dict(registered_model.aliases)

    return {
        "name": model_name,
        "description": registered_model.description or "",
        "versions": versions,
        "aliases": aliases,
    }


def compare_model_versions(
    model_name: str,
    version_a: str,
    version_b: str,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Compare metrics between two model versions.

    Args:
        model_name: Name of the registered model.
        version_a: First version (baseline).
        version_b: Second version (candidate).
        tracking_uri: Optional tracking URI override.

    Returns:
        Dictionary with metrics from both versions and their deltas.
    """
    client = get_mlflow_client(tracking_uri)

    mv_a = client.get_model_version(name=model_name, version=version_a)
    mv_b = client.get_model_version(name=model_name, version=version_b)

    run_a = client.get_run(mv_a.run_id)
    run_b = client.get_run(mv_b.run_id)

    metrics_a = dict(run_a.data.metrics)
    metrics_b = dict(run_b.data.metrics)

    common_keys = set(metrics_a.keys()) & set(metrics_b.keys())
    deltas = {key: metrics_b[key] - metrics_a[key] for key in common_keys}

    return {
        "version_a": {"version": version_a, "metrics": metrics_a},
        "version_b": {"version": version_b, "metrics": metrics_b},
        "deltas": deltas,
    }
