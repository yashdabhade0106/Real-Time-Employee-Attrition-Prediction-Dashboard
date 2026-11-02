"""Model evaluation utilities for attrition prediction.

Provides comprehensive model evaluation including metrics computation,
cross-validation, bootstrap confidence intervals, and calibration diagnostics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.utils import resample

from .logging_conf import get_logger

log = get_logger(__name__)

RANDOM_STATE: int = 42


def evaluate_model(
    model: Any,
    X_test: ArrayLike,
    y_test: ArrayLike,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate a trained model on a held-out test set.

    Args:
        model: A fitted estimator with ``predict_proba``.
        X_test: Test feature matrix.
        y_test: True binary labels (0/1).
        threshold: Decision threshold for converting probabilities to classes.

    Returns:
        Dictionary with accuracy, precision, recall, f1, roc_auc, pr_auc,
        balanced_accuracy, mcc, confusion_matrix, and threshold.
    """
    log.info("evaluate_model.start", threshold=threshold, n_samples=len(y_test))

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    results: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "mcc": float(matthews_corrcoef(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "threshold": threshold,
    }

    log.info(
        "evaluate_model.done",
        accuracy=round(results["accuracy"], 4),
        f1=round(results["f1"], 4),
        roc_auc=round(results["roc_auc"], 4),
    )
    return results


def cross_validate_model(
    model: Any,
    X: ArrayLike,
    y: ArrayLike,
    cv: int = 5,
    n_repeats: int = 3,
    random_state: int = RANDOM_STATE,
) -> dict[str, dict[str, float]]:
    """Perform repeated stratified K-fold cross-validation.

    Args:
        model: An unfitted estimator (will be cloned internally).
        X: Full feature matrix.
        y: Full binary labels.
        cv: Number of folds per repeat.
        n_repeats: Number of repetitions.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary mapping metric names to ``{"mean": float, "std": float}``
        sub-dicts.  Metrics: accuracy, precision, recall, f1, roc_auc,
        balanced_accuracy.
    """
    log.info(
        "cross_validate_model.start",
        cv=cv,
        n_repeats=n_repeats,
        n_samples=len(y),
    )

    rskf = RepeatedStratifiedKFold(
        n_splits=cv,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "balanced_accuracy": "balanced_accuracy",
    }

    cv_results = cross_validate(
        model,
        X,
        y,
        cv=rskf,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    summary: dict[str, dict[str, float]] = {}
    for metric_name in scoring:
        key = f"test_{metric_name}"
        values = cv_results[key]
        summary[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    log.info(
        "cross_validate_model.done",
        f1_mean=round(summary["f1"]["mean"], 4),
        roc_auc_mean=round(summary["roc_auc"]["mean"], 4),
    )
    return summary


def bootstrap_confidence_intervals(
    model: Any,
    X_test: ArrayLike,
    y_test: ArrayLike,
    threshold: float = 0.5,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: int = RANDOM_STATE,
) -> dict[str, dict[str, float]]:
    """Compute bootstrap confidence intervals for evaluation metrics.

    Args:
        model: A fitted estimator with ``predict_proba``.
        X_test: Test feature matrix.
        y_test: True binary labels.
        threshold: Decision threshold.
        n_bootstrap: Number of bootstrap iterations.
        ci: Confidence level (e.g. 0.95 for 95% CI).
        random_state: Random seed.

    Returns:
        Dictionary mapping metric names to
        ``{"mean": float, "lower": float, "upper": float}`` sub-dicts.
    """
    log.info(
        "bootstrap_ci.start",
        n_bootstrap=n_bootstrap,
        ci=ci,
        n_samples=len(y_test),
    )

    rng = np.random.RandomState(random_state)

    import pandas as pd

    X_arr = np.asarray(X_test) if not isinstance(X_test, pd.DataFrame) else X_test
    y_arr = np.asarray(y_test)

    metric_fns: dict[str, Any] = {
        "accuracy": lambda yt, yp, _: accuracy_score(yt, yp),
        "precision": lambda yt, yp, _: precision_score(yt, yp, zero_division=0),
        "recall": lambda yt, yp, _: recall_score(yt, yp, zero_division=0),
        "f1": lambda yt, yp, _: f1_score(yt, yp, zero_division=0),
        "roc_auc": lambda yt, _, ypr: roc_auc_score(yt, ypr),
        "balanced_accuracy": lambda yt, yp, _: balanced_accuracy_score(yt, yp),
        "mcc": lambda yt, yp, _: matthews_corrcoef(yt, yp),
    }

    boot_scores: dict[str, list[float]] = {k: [] for k in metric_fns}
    n = len(y_arr)

    for i in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        y_boot = y_arr[indices]

        # Skip if only one class present in bootstrap sample
        if len(np.unique(y_boot)) < 2:
            continue

        if isinstance(X_arr, pd.DataFrame):
            X_boot = X_arr.iloc[indices]
        else:
            X_boot = X_arr[indices]

        y_prob_boot = model.predict_proba(X_boot)[:, 1]
        y_pred_boot = (y_prob_boot >= threshold).astype(int)

        for name, fn in metric_fns.items():
            try:
                score = fn(y_boot, y_pred_boot, y_prob_boot)
                boot_scores[name].append(float(score))
            except ValueError:
                pass

    alpha = (1.0 - ci) / 2.0
    results: dict[str, dict[str, float]] = {}
    for name, scores in boot_scores.items():
        arr = np.array(scores)
        results[name] = {
            "mean": float(np.mean(arr)),
            "lower": float(np.percentile(arr, 100 * alpha)),
            "upper": float(np.percentile(arr, 100 * (1 - alpha))),
        }

    log.info(
        "bootstrap_ci.done",
        n_successful=len(boot_scores["f1"]),
        f1_mean=round(results["f1"]["mean"], 4),
    )
    return results


def compute_brier_score(
    y_true: ArrayLike,
    y_prob: ArrayLike,
) -> float:
    """Compute the Brier score (mean squared error of predicted probabilities).

    Lower is better; 0 is perfect calibration.

    Args:
        y_true: True binary labels (0/1).
        y_prob: Predicted probabilities for the positive class.

    Returns:
        Brier score as a float.
    """
    score = float(brier_score_loss(y_true, y_prob))
    log.debug("compute_brier_score", brier_score=round(score, 6))
    return score


def compute_ece(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    n_bins: int = 10,
) -> float:
    """Compute the Expected Calibration Error (ECE).

    ECE measures the gap between predicted confidence and actual accuracy,
    averaged across probability bins and weighted by bin size.

    Args:
        y_true: True binary labels (0/1).
        y_prob: Predicted probabilities for the positive class.
        n_bins: Number of equally-spaced probability bins.

    Returns:
        Expected Calibration Error as a float.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true_arr)

    for i in range(n_bins):
        mask = (y_prob_arr > bin_edges[i]) & (y_prob_arr <= bin_edges[i + 1])
        # Include exactly 0.0 in the first bin
        if i == 0:
            mask = mask | (y_prob_arr == bin_edges[i])

        bin_count = mask.sum()
        if bin_count == 0:
            continue

        bin_accuracy = y_true_arr[mask].mean()
        bin_confidence = y_prob_arr[mask].mean()
        ece += (bin_count / total) * abs(bin_accuracy - bin_confidence)

    log.debug("compute_ece", ece=round(ece, 6), n_bins=n_bins)
    return float(ece)


def generate_reliability_diagram_data(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    n_bins: int = 10,
) -> dict[str, list[float]]:
    """Generate data for a reliability (calibration) diagram.

    Args:
        y_true: True binary labels (0/1).
        y_prob: Predicted probabilities for the positive class.
        n_bins: Number of equally-spaced probability bins.

    Returns:
        Dictionary with:
        - ``mean_predicted_value``: average predicted probability per bin.
        - ``fraction_of_positives``: observed positive fraction per bin.
        - ``bin_counts``: number of samples per bin.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    mean_predicted: list[float] = []
    fraction_pos: list[float] = []
    counts: list[float] = []

    for i in range(n_bins):
        mask = (y_prob_arr > bin_edges[i]) & (y_prob_arr <= bin_edges[i + 1])
        if i == 0:
            mask = mask | (y_prob_arr == bin_edges[i])

        bin_count = int(mask.sum())
        if bin_count == 0:
            mean_predicted.append(float((bin_edges[i] + bin_edges[i + 1]) / 2))
            fraction_pos.append(0.0)
            counts.append(0.0)
        else:
            mean_predicted.append(float(y_prob_arr[mask].mean()))
            fraction_pos.append(float(y_true_arr[mask].mean()))
            counts.append(float(bin_count))

    log.debug(
        "generate_reliability_diagram_data",
        n_bins=n_bins,
        total_samples=int(sum(counts)),
    )

    return {
        "mean_predicted_value": mean_predicted,
        "fraction_of_positives": fraction_pos,
        "bin_counts": counts,
    }
