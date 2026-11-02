"""Prediction utilities for the attrition model.

Handles model loading, single-employee prediction, and batch inference
with risk-band assignment.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .logging_conf import get_logger

log = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Default risk bands (overridden by metadata if available)
_DEFAULT_RISK_BANDS: dict[str, dict[str, float]] = {
    "low": {"upper": 0.15},
    "moderate": {"lower": 0.15, "upper": 0.40},
    "high": {"lower": 0.40},
}


def load_model(
    version: str | None = None,
    model_dir: str = "models",
) -> tuple[Pipeline, dict[str, Any]]:
    """Load a serialised model pipeline and its metadata.

    If *version* is ``None``, the ``active`` version is read from
    ``models/config.json``.  Otherwise the specified version directory
    is loaded directly (e.g. ``"v1"``).

    Args:
        version: Model version string, or ``None`` for the active version.
        model_dir: Root directory containing versioned model folders.

    Returns:
        Tuple of ``(pipeline, metadata)`` where *pipeline* is the
        fitted scikit-learn ``Pipeline`` and *metadata* is a dict
        loaded from ``metadata.json``.

    Raises:
        FileNotFoundError: If the model directory or expected files are
            missing.
    """
    base = _PROJECT_ROOT / model_dir

    if version is None:
        config_path = base / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            version = config.get("active", "v1")
            log.info("load_model.resolved_active_version", version=version)
        else:
            version = "v1"
            log.warning("load_model.no_config_found, defaulting", version=version)

    version_dir = base / version
    pipeline_path = version_dir / "pipeline.joblib"
    metadata_path = version_dir / "metadata.json"

    if not pipeline_path.exists():
        raise FileNotFoundError(f"Pipeline not found at {pipeline_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")

    pipeline: Pipeline = joblib.load(pipeline_path)
    with open(metadata_path) as f:
        metadata: dict[str, Any] = json.load(f)

    log.info(
        "load_model.done",
        version=version,
        pipeline_path=str(pipeline_path),
    )
    return pipeline, metadata


def predict_single(
    pipeline: Pipeline,
    metadata: dict[str, Any],
    employee: dict[str, Any] | pd.DataFrame,
) -> dict[str, Any]:
    """Generate a prediction for a single employee.

    Args:
        pipeline: Fitted scikit-learn pipeline with ``predict_proba``.
        metadata: Model metadata dict (must contain ``threshold`` and
            optionally ``risk_bands`` and ``version``).
        employee: Employee feature data as a dict or single-row DataFrame.

    Returns:
        Dictionary with::

            {
                "class": 0 | 1,
                "probability": float,   # calibrated probability
                "risk_band": str,       # "Low" | "Moderate" | "High"
                "threshold": float,
                "model_version": str,
                "timestamp": str,       # ISO-8601
            }
    """
    log.info("predict_single.start")

    if isinstance(employee, dict):
        df = pd.DataFrame([employee])
    elif isinstance(employee, pd.DataFrame):
        df = employee.head(1).copy()
    else:
        raise TypeError(f"Expected dict or DataFrame, got {type(employee)}")

    threshold = metadata.get("threshold", 0.5)
    risk_bands = metadata.get("risk_bands", _DEFAULT_RISK_BANDS)
    model_version = metadata.get("version", "unknown")

    prob = float(pipeline.predict_proba(df)[:, 1][0])
    pred_class = int(prob >= threshold)
    band = _assign_risk_band(prob, risk_bands)

    result: dict[str, Any] = {
        "class": pred_class,
        "probability": round(prob, 4),
        "risk_band": band,
        "threshold": threshold,
        "model_version": model_version,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }

    log.info(
        "predict_single.done",
        probability=result["probability"],
        risk_band=band,
    )
    return result


def predict_batch(
    pipeline: Pipeline,
    metadata: dict[str, Any],
    employees_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate predictions for a batch of employees.

    Appends prediction columns to a *copy* of the input DataFrame.

    Args:
        pipeline: Fitted scikit-learn pipeline.
        metadata: Model metadata dict.
        employees_df: DataFrame where each row is an employee.

    Returns:
        Copy of *employees_df* with added columns: ``predicted_class``,
        ``attrition_probability``, ``risk_band``, ``model_version``,
        ``prediction_timestamp``.
    """
    log.info("predict_batch.start", n_employees=len(employees_df))

    threshold = metadata.get("threshold", 0.5)
    risk_bands = metadata.get("risk_bands", _DEFAULT_RISK_BANDS)
    model_version = metadata.get("version", "unknown")

    probs = pipeline.predict_proba(employees_df)[:, 1]

    result_df = employees_df.copy()
    result_df["attrition_probability"] = np.round(probs, 4)
    result_df["predicted_class"] = (probs >= threshold).astype(int)
    result_df["risk_band"] = [
        _assign_risk_band(p, risk_bands) for p in probs
    ]
    result_df["model_version"] = model_version
    result_df["prediction_timestamp"] = datetime.now(tz=timezone.utc).isoformat()

    log.info(
        "predict_batch.done",
        n_employees=len(result_df),
        n_high_risk=int((result_df["risk_band"] == "High").sum()),
    )
    return result_df


def _assign_risk_band(
    probability: float,
    risk_bands: dict[str, dict[str, float]],
) -> str:
    """Map a probability to a risk-band label.

    Args:
        probability: Calibrated attrition probability.
        risk_bands: Band definitions with ``upper`` / ``lower`` keys.

    Returns:
        One of ``"Low"``, ``"Moderate"``, or ``"High"``.
    """
    low_upper = risk_bands.get("low", {}).get("upper", 0.15)
    high_lower = risk_bands.get("high", {}).get("lower", 0.40)

    if probability <= low_upper:
        return "Low"
    elif probability <= high_lower:
        return "Moderate"
    else:
        return "High"
