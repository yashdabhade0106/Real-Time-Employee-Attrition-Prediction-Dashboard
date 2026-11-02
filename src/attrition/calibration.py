"""Probability calibration utilities for attrition prediction models.

Provides model calibration via CalibratedClassifierCV, automatic method
selection (sigmoid vs isotonic), and risk-band derivation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold

from .logging_conf import get_logger

log = get_logger(__name__)

RANDOM_STATE: int = 42


def calibrate_model(
    model: Any,
    X_cal: ArrayLike,
    y_cal: ArrayLike,
    method: str = "auto",
) -> CalibratedClassifierCV:
    """Calibrate a fitted model's probability estimates.

    When ``method="auto"``, both sigmoid (Platt scaling) and isotonic
    regression are tried and the one with the lower Brier score is selected.

    Args:
        model: A fitted estimator with ``predict_proba``.
        X_cal: Calibration feature matrix.
        y_cal: Calibration binary labels.
        method: Calibration method — ``"sigmoid"``, ``"isotonic"``, or
            ``"auto"`` (default).

    Returns:
        A fitted ``CalibratedClassifierCV`` instance.
    """
    log.info("calibrate_model.start", method=method, n_samples=len(y_cal))

    if method == "auto":
        best_method = select_calibration_method(model, X_cal, y_cal)
        log.info("calibrate_model.auto_selected", best_method=best_method)
    else:
        best_method = method

    calibrated = CalibratedClassifierCV(
        estimator=model,
        method=best_method,
        cv="prefit",
    )
    calibrated.fit(X_cal, y_cal)

    # Log post-calibration Brier score
    y_prob_cal = calibrated.predict_proba(X_cal)[:, 1]
    brier = float(brier_score_loss(y_cal, y_prob_cal))
    log.info("calibrate_model.done", method=best_method, brier_score=round(brier, 6))

    return calibrated


def select_calibration_method(
    model: Any,
    X: ArrayLike,
    y: ArrayLike,
    cv: int = 5,
) -> str:
    """Select the best calibration method via cross-validation.

    Compares sigmoid (Platt scaling) and isotonic regression using
    stratified K-fold cross-validation and picks the method with the
    lower mean Brier score.

    Args:
        model: A fitted estimator with ``predict_proba``.
        X: Feature matrix.
        y: Binary labels.
        cv: Number of cross-validation folds.

    Returns:
        Name of the best method: ``"sigmoid"`` or ``"isotonic"``.
    """
    log.info("select_calibration_method.start", cv=cv)

    y_arr = np.asarray(y)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    method_scores: dict[str, list[float]] = {
        "sigmoid": [],
        "isotonic": [],
    }

    import pandas as pd

    X_is_df = isinstance(X, pd.DataFrame)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_arr)):
        if X_is_df:
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
        else:
            X_arr = np.asarray(X)
            X_train_fold = X_arr[train_idx]
            X_val_fold = X_arr[val_idx]

        y_train_fold = y_arr[train_idx]
        y_val_fold = y_arr[val_idx]

        for method_name in method_scores:
            try:
                cal = CalibratedClassifierCV(
                    estimator=model,
                    method=method_name,
                    cv="prefit",
                )
                cal.fit(X_train_fold, y_train_fold)
                y_prob_val = cal.predict_proba(X_val_fold)[:, 1]
                brier = float(brier_score_loss(y_val_fold, y_prob_val))
                method_scores[method_name].append(brier)
            except Exception as exc:
                log.warning(
                    "select_calibration_method.fold_error",
                    method=method_name,
                    fold=fold_idx,
                    error=str(exc),
                )

    results: dict[str, float] = {}
    for method_name, scores in method_scores.items():
        mean_brier = float(np.mean(scores)) if scores else float("inf")
        results[method_name] = mean_brier
        log.debug(
            "select_calibration_method.result",
            method=method_name,
            mean_brier=round(mean_brier, 6),
        )

    best = min(results, key=lambda k: results[k])
    log.info(
        "select_calibration_method.done",
        best_method=best,
        sigmoid_brier=round(results.get("sigmoid", float("inf")), 6),
        isotonic_brier=round(results.get("isotonic", float("inf")), 6),
    )
    return best


def derive_risk_bands(
    y_true: ArrayLike,
    calibrated_probs: ArrayLike,
    method: str = "precision_targets",
) -> dict[str, Any]:
    """Derive risk-band cut-offs from calibrated probabilities.

    Two methods are supported:

    - ``"terciles"``: places thresholds at the 33rd and 66th percentiles
      of the positive-class predicted probabilities.
    - ``"precision_targets"``: selects thresholds so that the observed
      attrition rate within each band aligns with target ranges
      (Low < 15 %, Moderate 15–40 %, High > 40 %).

    Args:
        y_true: True binary labels (0/1).
        calibrated_probs: Calibrated predicted probabilities.
        method: ``"terciles"`` or ``"precision_targets"`` (default).

    Returns:
        Dictionary describing the three bands::

            {
                "low":      {"upper": float, "observed_rate": float},
                "moderate": {"lower": float, "upper": float, "observed_rate": float},
                "high":     {"lower": float, "observed_rate": float},
                "method":   str,
            }
    """
    log.info("derive_risk_bands.start", method=method, n_samples=len(y_true))

    y_arr = np.asarray(y_true, dtype=float)
    p_arr = np.asarray(calibrated_probs, dtype=float)

    if method == "terciles":
        bands = _derive_tercile_bands(y_arr, p_arr)
    elif method == "precision_targets":
        bands = _derive_precision_target_bands(y_arr, p_arr)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'terciles' or 'precision_targets'.")

    bands["method"] = method

    log.info(
        "derive_risk_bands.done",
        method=method,
        low_upper=bands["low"]["upper"],
        high_lower=bands["high"]["lower"],
    )
    return bands


def _derive_tercile_bands(
    y_true: np.ndarray,
    probs: np.ndarray,
) -> dict[str, Any]:
    """Derive risk bands using terciles of positive-class probabilities."""
    pos_probs = probs[y_true == 1]
    if len(pos_probs) == 0:
        pos_probs = probs

    lower_cut = float(np.percentile(pos_probs, 33))
    upper_cut = float(np.percentile(pos_probs, 66))

    low_mask = probs <= lower_cut
    mod_mask = (probs > lower_cut) & (probs <= upper_cut)
    high_mask = probs > upper_cut

    return {
        "low": {
            "upper": round(lower_cut, 4),
            "observed_rate": _safe_rate(y_true, low_mask),
        },
        "moderate": {
            "lower": round(lower_cut, 4),
            "upper": round(upper_cut, 4),
            "observed_rate": _safe_rate(y_true, mod_mask),
        },
        "high": {
            "lower": round(upper_cut, 4),
            "observed_rate": _safe_rate(y_true, high_mask),
        },
    }


def _derive_precision_target_bands(
    y_true: np.ndarray,
    probs: np.ndarray,
) -> dict[str, Any]:
    """Derive risk bands using precision-target thresholding.

    Searches for thresholds where:
    - Low band: observed attrition rate < 15 %
    - Moderate band: observed attrition rate 15–40 %
    - High band: observed attrition rate > 40 %
    """
    best_low_upper = 0.15
    best_high_lower = 0.40

    # Grid search for optimal thresholds
    candidates = np.arange(0.05, 0.95, 0.01)

    best_score = float("inf")
    target_low_rate = 0.075  # midpoint of [0, 0.15]
    target_mod_rate = 0.275  # midpoint of [0.15, 0.40]
    target_high_rate = 0.60  # roughly above 0.40

    for low_t in candidates:
        for high_t in candidates:
            if high_t <= low_t + 0.05:
                continue

            low_mask = probs <= low_t
            mod_mask = (probs > low_t) & (probs <= high_t)
            high_mask = probs > high_t

            # Require minimum samples in each band
            if low_mask.sum() < 5 or mod_mask.sum() < 5 or high_mask.sum() < 5:
                continue

            low_rate = y_true[low_mask].mean()
            mod_rate = y_true[mod_mask].mean()
            high_rate = y_true[high_mask].mean()

            # Penalise deviation from target rate ranges
            score = 0.0
            if low_rate > 0.15:
                score += (low_rate - 0.15) ** 2 * 10
            if mod_rate < 0.15 or mod_rate > 0.40:
                score += min((mod_rate - 0.15) ** 2, (mod_rate - 0.40) ** 2) * 5
            if high_rate < 0.40:
                score += (0.40 - high_rate) ** 2 * 10

            # Also push towards monotonicity
            if not (low_rate <= mod_rate <= high_rate):
                score += 1.0

            if score < best_score:
                best_score = score
                best_low_upper = float(low_t)
                best_high_lower = float(high_t)

    low_mask = probs <= best_low_upper
    mod_mask = (probs > best_low_upper) & (probs <= best_high_lower)
    high_mask = probs > best_high_lower

    return {
        "low": {
            "upper": round(best_low_upper, 4),
            "observed_rate": _safe_rate(y_true, low_mask),
        },
        "moderate": {
            "lower": round(best_low_upper, 4),
            "upper": round(best_high_lower, 4),
            "observed_rate": _safe_rate(y_true, mod_mask),
        },
        "high": {
            "lower": round(best_high_lower, 4),
            "observed_rate": _safe_rate(y_true, high_mask),
        },
    }


def _safe_rate(y_true: np.ndarray, mask: np.ndarray) -> float:
    """Compute observed positive rate for a mask, returning 0.0 if empty."""
    if mask.sum() == 0:
        return 0.0
    return round(float(y_true[mask].mean()), 4)
