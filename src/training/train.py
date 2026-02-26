"""Training pipeline with MLflow experiment tracking.

Trains RandomForest and XGBoost models on sklearn datasets with
automatic parameter, metric, and artifact logging via MLflow.
"""

import platform
import time
from typing import Any

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.training.evaluate import evaluate_model
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

DATASET_LOADERS = {
    "iris": load_iris,
    "wine": load_wine,
}


def load_dataset(
    dataset_name: str, test_size: float = 0.2, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load and split a sklearn dataset.

    Args:
        dataset_name: Name of the dataset ('iris' or 'wine').
        test_size: Fraction of data to use for testing.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names).

    Raises:
        ValueError: If dataset_name is not supported.
    """
    if dataset_name not in DATASET_LOADERS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Choose from: {list(DATASET_LOADERS.keys())}"
        )

    data = DATASET_LOADERS[dataset_name]()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, list(data.feature_names)


def _get_run_metadata() -> dict[str, str]:
    """Collect run metadata for MLflow tagging.

    Returns:
        Dictionary of metadata tags.
    """
    metadata = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }

    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            metadata["git_commit"] = result.stdout.strip()
    except Exception:
        pass

    return metadata


def build_model(model_type: str, params: dict[str, Any]) -> Any:
    """Create a model instance from type and parameters.

    Args:
        model_type: Either 'random_forest' or 'xgboost'.
        params: Hyperparameters for the model constructor.

    Returns:
        Instantiated sklearn-compatible model.

    Raises:
        ValueError: If model_type is not supported.
    """
    if model_type == "random_forest":
        return RandomForestClassifier(**params)
    elif model_type == "xgboost":
        return XGBClassifier(**params, eval_metric="mlogloss")
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_model(
    model_type: str = "random_forest",
    config_path: str | None = None,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Train a model and log everything to MLflow.

    Args:
        model_type: Type of model to train ('random_forest' or 'xgboost').
        config_path: Optional path to config YAML. Uses default if None.
        tracking_uri: Optional MLflow tracking URI override.

    Returns:
        Dictionary with run_id, model_type, and metrics.
    """
    config = load_config(config_path)
    training_config = config["training"]
    mlflow_config = config["mlflow"]

    uri = tracking_uri or mlflow_config["tracking_uri"]
    mlflow.set_tracking_uri(uri)

    experiment_name = mlflow_config["experiment_name"]
    mlflow.set_experiment(experiment_name)

    dataset_name = training_config["dataset"]
    X_train, X_test, y_train, y_test, feature_names = load_dataset(
        dataset_name=dataset_name,
        test_size=training_config["test_size"],
        random_state=training_config["random_state"],
    )

    model_params = dict(training_config["models"][model_type])
    model = build_model(model_type, model_params)

    with mlflow.start_run() as run:
        run_metadata = _get_run_metadata()
        for key, value in run_metadata.items():
            mlflow.set_tag(key, value)

        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("dataset", dataset_name)
        mlflow.log_params(model_params)

        logger.info(
            "Training %s on %s dataset (train=%d, test=%d)",
            model_type,
            dataset_name,
            len(X_train),
            len(X_test),
        )

        start_time = time.time()
        model.fit(X_train, y_train)
        training_duration = time.time() - start_time

        mlflow.log_metric("training_duration_seconds", training_duration)

        metrics = evaluate_model(model, X_test, y_test)

        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(metric_name, metric_value)

        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train_samples", X_train.shape[0])
        mlflow.log_param("n_test_samples", X_test.shape[0])

        input_example = X_test[:1]

        if model_type == "xgboost":
            mlflow.xgboost.log_model(model, "model", input_example=input_example)
        else:
            mlflow.sklearn.log_model(model, "model", input_example=input_example)

        logger.info(
            "Run %s complete — accuracy=%.4f, duration=%.2fs",
            run.info.run_id,
            metrics["accuracy"],
            training_duration,
        )

        return {
            "run_id": run.info.run_id,
            "model_type": model_type,
            "metrics": metrics,
            "training_duration": training_duration,
        }


def train_all_models(
    config_path: str | None = None,
    tracking_uri: str | None = None,
) -> list[dict[str, Any]]:
    """Train all configured model types and return their results.

    Args:
        config_path: Optional path to config YAML.
        tracking_uri: Optional MLflow tracking URI override.

    Returns:
        List of result dictionaries from each training run.
    """
    config = load_config(config_path)
    model_types = list(config["training"]["models"].keys())
    results = []

    for model_type in model_types:
        logger.info("Training model: %s", model_type)
        result = train_model(
            model_type=model_type,
            config_path=config_path,
            tracking_uri=tracking_uri,
        )
        results.append(result)

    return results
