"""Model evaluation utilities.

Computes classification metrics including accuracy, precision, recall, F1 score,
and confusion matrix for trained models.
"""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Evaluate a trained model on test data.

    Args:
        model: Trained sklearn-compatible classifier with a predict method.
        X_test: Test feature matrix.
        y_test: True labels for the test set.

    Returns:
        Dictionary containing accuracy, precision, recall, f1, and confusion_matrix.
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    logger.info(
        "Evaluation — accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
        accuracy,
        precision,
        recall,
        f1,
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
    }


def compare_metrics(
    metrics_a: dict[str, Any],
    metrics_b: dict[str, Any],
) -> dict[str, float]:
    """Compare two sets of metrics and return the differences.

    Args:
        metrics_a: Metrics dictionary from the first model (baseline).
        metrics_b: Metrics dictionary from the second model (candidate).

    Returns:
        Dictionary of metric deltas (positive means B is better).
    """
    comparable_keys = ["accuracy", "precision", "recall", "f1"]
    deltas = {}

    for key in comparable_keys:
        if key in metrics_a and key in metrics_b:
            deltas[f"{key}_delta"] = metrics_b[key] - metrics_a[key]

    return deltas
