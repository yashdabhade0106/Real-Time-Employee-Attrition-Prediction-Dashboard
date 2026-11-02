"""Report generation for employee attrition predictions.

Produces styled HTML reports for individual employees and aggregate
team-level analytics with privacy-preserving group suppression.
"""

from __future__ import annotations

import html
from typing import Any

import pandas as pd

from .logging_conf import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Disclaimer text (legally required)
# ---------------------------------------------------------------------------
_DISCLAIMER = (
    "This is an estimate, not a guarantee. This model should never be "
    "the sole basis for an employment decision."
)

# ---------------------------------------------------------------------------
# CSS for the HTML report
# ---------------------------------------------------------------------------
_REPORT_CSS = """\
<style>
  body { font-family: 'Segoe UI', Arial, sans-serif; margin: 2rem; color: #1a1a2e; }
  .report-container { max-width: 800px; margin: auto; }
  h1 { color: #16213e; border-bottom: 3px solid #0f3460; padding-bottom: .5rem; }
  h2 { color: #0f3460; margin-top: 1.5rem; }
  .risk-badge { display: inline-block; padding: 4px 16px; border-radius: 12px;
                font-weight: bold; font-size: 1.1rem; }
  .risk-high   { background: #ff6b6b; color: #fff; }
  .risk-medium { background: #ffa502; color: #fff; }
  .risk-low    { background: #2ed573; color: #fff; }
  table { width: 100%%; border-collapse: collapse; margin: 1rem 0; }
  th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid #dfe6e9; }
  th { background: #0f3460; color: #fff; }
  tr:nth-child(even) { background: #f1f2f6; }
  .disclaimer { background: #fff3cd; border: 1px solid #ffc107;
                padding: 12px 16px; border-radius: 6px; margin-top: 2rem;
                font-size: 0.9rem; }
  .positive { color: #e74c3c; }
  .negative { color: #27ae60; }
  .shap-bar { height: 14px; border-radius: 3px; display: inline-block; }
  .shap-bar-pos { background: #e74c3c; }
  .shap-bar-neg { background: #27ae60; }
  .meta { color: #636e72; font-size: 0.85rem; }
</style>
"""


def _risk_badge_html(risk_band: str) -> str:
    """Return styled HTML badge for a risk band."""
    css_class = {
        "High": "risk-high",
        "Medium": "risk-medium",
        "Low": "risk-low",
    }.get(risk_band, "risk-medium")
    safe = html.escape(risk_band)
    return f'<span class="risk-badge {css_class}">{safe}</span>'


def _shap_bar_html(value: float, max_abs: float) -> str:
    """Render a tiny horizontal bar proportional to the SHAP value."""
    if max_abs == 0:
        return ""
    width_pct = min(abs(value) / max_abs * 100, 100)
    css_class = "shap-bar-pos" if value > 0 else "shap-bar-neg"
    return f'<span class="shap-bar {css_class}" style="width:{width_pct:.0f}%"></span>'


def generate_employee_report_html(
    employee: dict[str, Any],
    prediction: dict[str, Any],
    insights: dict[str, Any],
    shap_factors: list[dict[str, Any]],
) -> str:
    """Generate a styled HTML report for a single employee prediction.

    Args:
        employee: Dictionary of employee feature values (e.g. Age,
            Department, MonthlyIncome).  Values are HTML-escaped.
        prediction: Prediction output with keys ``predicted_class``,
            ``probability``, and ``risk_band``.
        insights: Free-form insights dict with key ``recommendations``
            (list of strings) and optional ``summary`` (string).
        shap_factors: List of dicts with ``feature``, ``value`` (SHAP),
            and ``direction`` (+1 / -1) sorted by absolute importance.

    Returns:
        Complete HTML document string.
    """
    log.info("generating_employee_report")

    risk_band: str = prediction.get("risk_band", "Unknown")
    probability: float = prediction.get("probability", 0.0)

    # ---- Header & risk section ----
    parts: list[str] = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>Employee Attrition Risk Report</title>{_REPORT_CSS}</head>",
        "<body><div class='report-container'>",
        "<h1>Employee Attrition Risk Report</h1>",
        f"<h2>Risk Assessment: {_risk_badge_html(risk_band)}</h2>",
        f"<p>Predicted probability of attrition: <strong>{probability:.1%}</strong></p>",
    ]

    # ---- Employee profile table ----
    parts.append("<h2>Employee Profile</h2><table><tr><th>Feature</th><th>Value</th></tr>")
    for key, val in employee.items():
        safe_key = html.escape(str(key))
        safe_val = html.escape(str(val))
        parts.append(f"<tr><td>{safe_key}</td><td>{safe_val}</td></tr>")
    parts.append("</table>")

    # ---- SHAP factors table ----
    if shap_factors:
        max_abs = max((abs(f.get("value", 0)) for f in shap_factors), default=1.0)
        parts.append(
            "<h2>Top Contributing Factors</h2>"
            "<table><tr><th>Feature</th><th>Impact</th><th>SHAP Value</th><th></th></tr>"
        )
        for factor in shap_factors:
            fname = html.escape(str(factor.get("feature", "")))
            fval = factor.get("value", 0.0)
            direction = "positive" if fval > 0 else "negative"
            label = "↑ increases risk" if fval > 0 else "↓ decreases risk"
            bar = _shap_bar_html(fval, max_abs)
            parts.append(
                f"<tr><td>{fname}</td>"
                f'<td class="{direction}">{label}</td>'
                f"<td>{fval:+.4f}</td><td>{bar}</td></tr>"
            )
        parts.append("</table>")

    # ---- Insights / recommendations ----
    summary = insights.get("summary", "")
    recommendations = insights.get("recommendations", [])

    if summary:
        parts.append(f"<h2>Summary</h2><p>{html.escape(summary)}</p>")

    if recommendations:
        parts.append("<h2>Recommendations</h2><ul>")
        for rec in recommendations:
            parts.append(f"<li>{html.escape(str(rec))}</li>")
        parts.append("</ul>")

    # ---- Disclaimer ----
    parts.append(f'<div class="disclaimer">⚠️ <strong>Disclaimer:</strong> {_DISCLAIMER}</div>')
    parts.append("</div></body></html>")

    report_html = "\n".join(parts)
    log.info("employee_report_generated", risk_band=risk_band)
    return report_html


def generate_team_analytics(
    predictions_df: pd.DataFrame,
    min_group_size: int = 5,
) -> dict[str, Any]:
    """Aggregate attrition analytics by department, job role, etc.

    Groups smaller than *min_group_size* are suppressed to protect
    employee privacy (k-anonymity principle).

    Args:
        predictions_df: DataFrame with at least columns ``Department``,
            ``JobRole``, ``probability``, ``risk_band``.  Additional
            grouping columns (``Gender``, ``MaritalStatus``) are used
            if present.
        min_group_size: Minimum number of employees in a group for the
            group to be included in the output.  Groups below this
            threshold are replaced with a ``"<suppressed>"`` placeholder.

    Returns:
        Dict with keys for each grouping dimension, each mapping to a
        list of group-level summary dicts with ``group``, ``count``,
        ``mean_probability``, ``high_risk_count``, ``high_risk_pct``.
    """
    log.info(
        "generating_team_analytics",
        n_rows=len(predictions_df),
        min_group_size=min_group_size,
    )

    grouping_columns = [
        col for col in ["Department", "JobRole", "Gender", "MaritalStatus"]
        if col in predictions_df.columns
    ]

    result: dict[str, Any] = {"suppressed_groups": {}}

    for col in grouping_columns:
        group_stats: list[dict[str, Any]] = []
        suppressed_count = 0

        grouped = predictions_df.groupby(col, observed=True)
        for group_name, group_df in grouped:
            if len(group_df) < min_group_size:
                suppressed_count += 1
                log.debug(
                    "group_suppressed",
                    column=col,
                    group=str(group_name),
                    size=len(group_df),
                )
                continue

            high_risk = group_df[
                group_df["risk_band"].str.lower() == "high"
            ] if "risk_band" in group_df.columns else pd.DataFrame()

            prob_col = (
                "probability" if "probability" in group_df.columns else None
            )

            stats: dict[str, Any] = {
                "group": str(group_name),
                "count": int(len(group_df)),
                "mean_probability": (
                    round(float(group_df[prob_col].mean()), 4)
                    if prob_col else None
                ),
                "high_risk_count": int(len(high_risk)),
                "high_risk_pct": (
                    round(len(high_risk) / len(group_df) * 100, 2)
                    if len(group_df) > 0 else 0.0
                ),
            }
            group_stats.append(stats)

        result[col] = group_stats
        result["suppressed_groups"][col] = suppressed_count

    log.info(
        "team_analytics_generated",
        dimensions=grouping_columns,
        suppressed=result["suppressed_groups"],
    )
    return result
