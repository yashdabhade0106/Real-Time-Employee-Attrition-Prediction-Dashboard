"""Fairness metrics and disparate impact analysis for attrition predictions.

Uses fairlearn's MetricFrame to compute group-level metrics and detect
disparities that violate the 80-percent rule (disparate impact ratio).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.metrics import (
    false_positive_rate,
    precision_score,
    recall_score,
)

from .logging_conf import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default metric functions recognised by this module
# ---------------------------------------------------------------------------
_METRIC_FUNCS: dict[str, Any] = {
    "recall": recall_score,
    "precision": precision_score,
    "selection_rate": selection_rate,
    "false_positive_rate": false_positive_rate,
}


def compute_fairness_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    sensitive_features: pd.DataFrame,
    metrics: list[str] | None = None,
) -> dict[str, Any]:
    """Compute group-level fairness metrics across sensitive feature groups.

    Args:
        y_true: Ground-truth binary labels (0/1).
        y_pred: Predicted binary labels (0/1).
        sensitive_features: DataFrame where each column is a sensitive
            attribute (e.g. Gender, Department).  Only the **first** column
            is used for the primary analysis; pass a single-column frame
            for clarity.
        metrics: Subset of ``['recall', 'precision', 'selection_rate',
            'false_positive_rate']`` to compute.  ``None`` → all four.

    Returns:
        Dictionary with keys:

        * ``overall``  – dict of metric → float (overall score).
        * ``by_group`` – dict of metric → {group_label: float}.
        * ``disparity_ratios`` – dict of metric → float (min-group / max-group).
        * ``sensitive_feature`` – name of the sensitive attribute analysed.
    """
    requested = metrics or list(_METRIC_FUNCS.keys())
    funcs = {m: _METRIC_FUNCS[m] for m in requested if m in _METRIC_FUNCS}
    if not funcs:
        raise ValueError(
            f"No valid metrics in {requested}. "
            f"Choose from {list(_METRIC_FUNCS.keys())}."
        )

    feature_name: str = sensitive_features.columns[0]
    sf_series: pd.Series = sensitive_features.iloc[:, 0]

    log.info(
        "computing_fairness_metrics",
        sensitive_feature=feature_name,
        metrics=list(funcs.keys()),
        n_samples=len(y_true),
    )

    mf = MetricFrame(
        metrics=funcs,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sf_series,
    )

    overall: dict[str, float] = mf.overall.to_dict()
    by_group: dict[str, dict[str, float]] = {
        metric: mf.by_group[metric].to_dict() for metric in funcs
    }

    # Disparity ratios: min / max for each metric (bounded away from /0)
    disparity_ratios: dict[str, float] = {}
    for metric in funcs:
        values = np.array(list(by_group[metric].values()), dtype=float)
        valid = values[np.isfinite(values)]
        if len(valid) == 0 or valid.max() == 0.0:
            disparity_ratios[metric] = float("nan")
        else:
            disparity_ratios[metric] = float(valid.min() / valid.max())

    result: dict[str, Any] = {
        "overall": overall,
        "by_group": by_group,
        "disparity_ratios": disparity_ratios,
        "sensitive_feature": feature_name,
    }

    log.info(
        "fairness_metrics_computed",
        disparity_ratios=disparity_ratios,
    )
    return result


def generate_fairness_report(metrics: dict[str, Any], sensitive_feature_name: str) -> str:
    """Render a Markdown-formatted fairness report.

    Args:
        metrics: Output of :func:`compute_fairness_metrics`.
        sensitive_feature_name: Human-readable name of the sensitive
            attribute (used in the report title).

    Returns:
        Markdown string ready for display or file output.
    """
    lines: list[str] = [
        f"# Fairness Report – {sensitive_feature_name}",
        "",
        "## Overall Metrics",
        "",
    ]

    for metric, value in metrics.get("overall", {}).items():
        lines.append(f"- **{metric}**: {value:.4f}")
    lines.append("")

    # Per-group table
    by_group: dict[str, dict[str, float]] = metrics.get("by_group", {})
    if by_group:
        first_metric = next(iter(by_group))
        groups = list(by_group[first_metric].keys())

        header = "| Group | " + " | ".join(by_group.keys()) + " |"
        separator = "| --- | " + " | ".join(["---"] * len(by_group)) + " |"
        lines.extend(["## Per-Group Metrics", "", header, separator])

        for group in groups:
            row_values = [
                f"{by_group[m].get(group, float('nan')):.4f}" for m in by_group
            ]
            lines.append(f"| {group} | " + " | ".join(row_values) + " |")
        lines.append("")

    # Disparity ratios
    disparity_ratios = metrics.get("disparity_ratios", {})
    if disparity_ratios:
        lines.extend(["## Disparity Ratios (min / max)", ""])
        for metric, ratio in disparity_ratios.items():
            flag = " ⚠️" if np.isfinite(ratio) and ratio < 0.8 else ""
            lines.append(f"- **{metric}**: {ratio:.4f}{flag}")
        lines.append("")

    report = "\n".join(lines)
    log.info("fairness_report_generated", sensitive_feature=sensitive_feature_name)
    return report


def check_disparities(
    metrics: dict[str, Any],
    threshold: float = 0.8,
) -> list[dict[str, Any]]:
    """Identify metrics that violate the 80-percent (disparate impact) rule.

    The *disparate impact ratio* for a metric is ``min_group / max_group``.
    A ratio below *threshold* signals a potential disparity.

    Args:
        metrics: Output of :func:`compute_fairness_metrics`.
        threshold: Minimum acceptable ratio (default 0.8 per the
            four-fifths / 80 %% rule).

    Returns:
        List of dicts, each with keys ``metric``, ``ratio``,
        ``threshold``, and ``groups`` (the by-group breakdown).
    """
    alerts: list[dict[str, Any]] = []
    disparity_ratios = metrics.get("disparity_ratios", {})
    by_group = metrics.get("by_group", {})

    for metric, ratio in disparity_ratios.items():
        if not np.isfinite(ratio):
            continue
        if ratio < threshold:
            alert = {
                "metric": metric,
                "ratio": round(ratio, 4),
                "threshold": threshold,
                "groups": by_group.get(metric, {}),
            }
            alerts.append(alert)
            log.warning(
                "disparity_detected",
                metric=metric,
                ratio=round(ratio, 4),
                threshold=threshold,
            )

    if not alerts:
        log.info("no_disparities_detected", threshold=threshold)

    return alerts
