"""Data and prediction drift monitoring utilities.

Provides Population Stability Index (PSI), Kolmogorov-Smirnov statistic,
and alert generation for feature-level and prediction-level drift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .logging_conf import get_logger

log = get_logger(__name__)

# Small constant to avoid log(0) in PSI calculation
_EPS: float = 1e-8


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute the Population Stability Index between two distributions.

    The PSI quantifies how much a variable's distribution has shifted
    between *reference* (training / baseline) and *current* (production)
    data.

    Interpretation:
        * PSI < 0.10 → no significant shift
        * 0.10 ≤ PSI < 0.20 → moderate shift
        * PSI ≥ 0.20 → significant shift

    Args:
        reference: 1-D array of reference (baseline) values.
        current: 1-D array of current (production) values.
        n_bins: Number of equal-width bins derived from the reference
            distribution.

    Returns:
        PSI value (non-negative float).
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    # Build bins from reference distribution (+ boundary padding)
    breakpoints = np.linspace(
        min(reference.min(), current.min()) - _EPS,
        max(reference.max(), current.max()) + _EPS,
        n_bins + 1,
    )

    ref_counts = np.histogram(reference, bins=breakpoints)[0].astype(float)
    cur_counts = np.histogram(current, bins=breakpoints)[0].astype(float)

    # Convert to proportions and clip to avoid log(0)
    ref_pct = np.clip(ref_counts / ref_counts.sum(), _EPS, None)
    cur_pct = np.clip(cur_counts / cur_counts.sum(), _EPS, None)

    psi_value: float = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi_value


def compute_ks_statistic(
    reference: np.ndarray,
    current: np.ndarray,
) -> float:
    """Compute the Kolmogorov-Smirnov statistic between two samples.

    The KS statistic is the maximum absolute difference between the
    empirical CDFs of *reference* and *current*.

    Args:
        reference: 1-D array of reference values.
        current: 1-D array of current values.

    Returns:
        KS statistic (float in [0, 1]).
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    ks_stat: float = float(sp_stats.ks_2samp(reference, current).statistic)
    return ks_stat


def compute_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_features: list[str],
) -> dict[str, dict[str, float]]:
    """Compute PSI and KS drift statistics for each numeric feature.

    Args:
        reference_df: Reference (training) DataFrame.
        current_df: Current (production) DataFrame.
        numeric_features: Column names to evaluate.

    Returns:
        Dict mapping each feature name to ``{"psi": float, "ks": float}``.
    """
    log.info(
        "computing_feature_drift",
        n_features=len(numeric_features),
        ref_rows=len(reference_df),
        cur_rows=len(current_df),
    )

    drift: dict[str, dict[str, float]] = {}
    for feature in numeric_features:
        if feature not in reference_df.columns or feature not in current_df.columns:
            log.warning("feature_missing_in_dataframe", feature=feature)
            continue

        ref_vals = reference_df[feature].dropna().values
        cur_vals = current_df[feature].dropna().values

        if len(ref_vals) == 0 or len(cur_vals) == 0:
            log.warning("empty_feature_values", feature=feature)
            continue

        psi = compute_psi(ref_vals, cur_vals)
        ks = compute_ks_statistic(ref_vals, cur_vals)
        drift[feature] = {"psi": round(psi, 6), "ks": round(ks, 6)}

    log.info("feature_drift_computed", features_evaluated=len(drift))
    return drift


def compute_prediction_drift(
    reference_probs: np.ndarray,
    current_probs: np.ndarray,
) -> dict[str, float]:
    """Compute drift metrics on predicted probabilities.

    Args:
        reference_probs: 1-D array of reference predicted probabilities.
        current_probs: 1-D array of current predicted probabilities.

    Returns:
        Dict with ``psi``, ``ks``, ``ref_mean``, ``cur_mean``,
        ``ref_std``, ``cur_std``.
    """
    reference_probs = np.asarray(reference_probs, dtype=float)
    current_probs = np.asarray(current_probs, dtype=float)

    log.info(
        "computing_prediction_drift",
        ref_samples=len(reference_probs),
        cur_samples=len(current_probs),
    )

    psi = compute_psi(reference_probs, current_probs)
    ks = compute_ks_statistic(reference_probs, current_probs)

    result: dict[str, float] = {
        "psi": round(psi, 6),
        "ks": round(ks, 6),
        "ref_mean": round(float(reference_probs.mean()), 6),
        "cur_mean": round(float(current_probs.mean()), 6),
        "ref_std": round(float(reference_probs.std()), 6),
        "cur_std": round(float(current_probs.std()), 6),
    }

    log.info("prediction_drift_computed", **result)
    return result


def check_drift_alerts(
    drift_metrics: dict[str, dict[str, float]],
    psi_threshold: float = 0.2,
    ks_threshold: float = 0.1,
) -> list[dict[str, object]]:
    """Generate alerts for features whose drift exceeds thresholds.

    Args:
        drift_metrics: Output of :func:`compute_feature_drift` – a dict
            mapping feature names to ``{"psi": float, "ks": float}``.
        psi_threshold: PSI value above which an alert is raised
            (default 0.2 = significant shift).
        ks_threshold: KS statistic above which an alert is raised
            (default 0.1).

    Returns:
        List of alert dicts with ``feature``, ``metric``, ``value``,
        ``threshold``, and ``severity`` (``"warning"`` or ``"critical"``).
    """
    alerts: list[dict[str, object]] = []

    for feature, values in drift_metrics.items():
        psi = values.get("psi", 0.0)
        ks = values.get("ks", 0.0)

        if psi > psi_threshold:
            severity = "critical" if psi > 2 * psi_threshold else "warning"
            alert = {
                "feature": feature,
                "metric": "psi",
                "value": psi,
                "threshold": psi_threshold,
                "severity": severity,
            }
            alerts.append(alert)
            log.warning(
                "drift_alert",
                feature=feature,
                metric="psi",
                value=psi,
                severity=severity,
            )

        if ks > ks_threshold:
            severity = "critical" if ks > 2 * ks_threshold else "warning"
            alert = {
                "feature": feature,
                "metric": "ks",
                "value": ks,
                "threshold": ks_threshold,
                "severity": severity,
            }
            alerts.append(alert)
            log.warning(
                "drift_alert",
                feature=feature,
                metric="ks",
                value=ks,
                severity=severity,
            )

    if not alerts:
        log.info(
            "no_drift_alerts",
            psi_threshold=psi_threshold,
            ks_threshold=ks_threshold,
        )

    return alerts
