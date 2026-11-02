"""SHAP-based explainability utilities for attrition predictions.

Provides local and global feature-importance computation using SHAP
(SHapley Additive exPlanations).  Automatically selects the appropriate
SHAP explainer based on the underlying model type.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from .logging_conf import get_logger

log = get_logger(__name__)


def compute_shap_values(
    pipeline: Pipeline,
    X: pd.DataFrame,
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, float, list[str]]:
    """Compute SHAP values for a set of observations.

    Auto-detects the model type inside the pipeline:

    - ``LinearExplainer`` for logistic regression.
    - ``TreeExplainer`` for tree-based models (RF, GBM, XGBoost, LightGBM).
    - Falls back to ``KernelExplainer`` otherwise.

    Args:
        pipeline: A fitted scikit-learn ``Pipeline`` whose last step is
            the estimator.
        X: Feature DataFrame (raw features — the pipeline's
            preprocessor is applied internally).
        feature_names: Optional explicit feature name list.  If ``None``,
            column names are taken from *X*.

    Returns:
        Tuple of ``(shap_values, expected_value, feature_names)`` where
        *shap_values* has shape ``(n_samples, n_features)`` and
        *expected_value* is the base value (log-odds or probability
        depending on the explainer).
    """
    log.info("compute_shap_values.start", n_samples=len(X))

    if feature_names is None:
        feature_names = list(X.columns)

    # Extract preprocessor and estimator from pipeline
    estimator = pipeline.steps[-1][1]
    if len(pipeline.steps) > 1:
        preprocessor = Pipeline(pipeline.steps[:-1])
        X_transformed = preprocessor.transform(X)
    else:
        X_transformed = X

    # Convert sparse to dense if needed
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    # Resolve the inner estimator for calibrated classifiers
    inner_estimator = _unwrap_estimator(estimator)

    # Build transformed feature names
    if len(pipeline.steps) > 1:
        transformed_names = _get_transformed_feature_names(preprocessor)
    else:
        transformed_names = feature_names

    # Select explainer
    model_type = type(inner_estimator).__name__
    log.info("compute_shap_values.model_type", model_type=model_type)

    if _is_linear(inner_estimator):
        log.info("compute_shap_values.using_linear_explainer")
        explainer = shap.LinearExplainer(inner_estimator, X_transformed)
        sv = explainer.shap_values(X_transformed)
        expected = float(explainer.expected_value)

    elif _is_tree_based(inner_estimator):
        log.info("compute_shap_values.using_tree_explainer")
        explainer = shap.TreeExplainer(inner_estimator)
        sv = explainer.shap_values(X_transformed)
        # For binary classifiers TreeExplainer may return a list of arrays
        if isinstance(sv, list):
            sv = sv[1]  # positive class
        ev = explainer.expected_value
        expected = float(ev[1]) if isinstance(ev, (list, np.ndarray)) else float(ev)

    else:
        log.info("compute_shap_values.using_kernel_explainer")
        # Sample background for KernelExplainer
        if len(X_transformed) > 100:
            background = shap.sample(X_transformed, 100)
        else:
            background = X_transformed
        explainer = shap.KernelExplainer(inner_estimator.predict_proba, background)
        sv = explainer.shap_values(X_transformed)
        if isinstance(sv, list):
            sv = sv[1]
        ev = explainer.expected_value
        expected = float(ev[1]) if isinstance(ev, (list, np.ndarray)) else float(ev)

    shap_values = np.asarray(sv)

    log.info(
        "compute_shap_values.done",
        shap_shape=shap_values.shape,
        n_features=len(transformed_names),
    )
    return shap_values, expected, transformed_names


def get_top_factors(
    shap_values_row: np.ndarray,
    feature_names: list[str],
    n_top: int = 10,
) -> list[dict[str, Any]]:
    """Extract the top contributing factors for a single prediction.

    Args:
        shap_values_row: 1-D array of SHAP values for one observation.
        feature_names: Feature names aligned with *shap_values_row*.
        n_top: Number of top factors to return.

    Returns:
        List of dicts sorted by absolute contribution (descending)::

            [
                {
                    "feature": str,
                    "contribution": float,
                    "direction": "increases risk" | "decreases risk",
                },
                ...
            ]
    """
    sv = np.asarray(shap_values_row).ravel()
    if len(sv) != len(feature_names):
        log.warning(
            "get_top_factors.length_mismatch",
            shap_len=len(sv),
            names_len=len(feature_names),
        )
        # Truncate to shorter
        n = min(len(sv), len(feature_names))
        sv = sv[:n]
        feature_names = feature_names[:n]

    abs_vals = np.abs(sv)
    top_indices = np.argsort(abs_vals)[::-1][:n_top]

    factors: list[dict[str, Any]] = []
    for idx in top_indices:
        factors.append(
            {
                "feature": feature_names[idx],
                "contribution": round(float(sv[idx]), 6),
                "direction": "increases risk" if sv[idx] > 0 else "decreases risk",
            }
        )

    log.debug("get_top_factors", n_top=n_top, top_feature=factors[0]["feature"] if factors else None)
    return factors


def get_global_feature_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> list[dict[str, float]]:
    """Compute global feature importance via mean absolute SHAP values.

    Args:
        shap_values: SHAP values array of shape ``(n_samples, n_features)``.
        feature_names: Feature names aligned with the columns of
            *shap_values*.

    Returns:
        List of dicts sorted by importance (descending)::

            [{"feature": str, "importance": float}, ...]
    """
    sv = np.asarray(shap_values)
    mean_abs = np.mean(np.abs(sv), axis=0)

    importance_list: list[dict[str, float]] = []
    for idx in np.argsort(mean_abs)[::-1]:
        importance_list.append(
            {
                "feature": feature_names[idx],
                "importance": round(float(mean_abs[idx]), 6),
            }
        )

    log.info(
        "get_global_feature_importance",
        n_features=len(importance_list),
        top_feature=importance_list[0]["feature"] if importance_list else None,
    )
    return importance_list


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _unwrap_estimator(estimator: Any) -> Any:
    """Unwrap a CalibratedClassifierCV to its base estimator."""
    from sklearn.calibration import CalibratedClassifierCV

    if isinstance(estimator, CalibratedClassifierCV):
        # Access the base estimator from the first calibrated classifier
        if hasattr(estimator, "calibrated_classifiers_"):
            inner = estimator.calibrated_classifiers_[0].estimator
            log.debug("_unwrap_estimator.unwrapped", inner_type=type(inner).__name__)
            return inner
        if hasattr(estimator, "estimator"):
            return estimator.estimator
    return estimator


def _is_linear(estimator: Any) -> bool:
    """Check if an estimator is a linear model."""
    linear_types = (
        "LogisticRegression",
        "SGDClassifier",
        "RidgeClassifier",
        "LinearSVC",
    )
    return type(estimator).__name__ in linear_types


def _is_tree_based(estimator: Any) -> bool:
    """Check if an estimator is a tree-based model."""
    tree_types = (
        "DecisionTreeClassifier",
        "RandomForestClassifier",
        "GradientBoostingClassifier",
        "XGBClassifier",
        "LGBMClassifier",
        "CatBoostClassifier",
        "ExtraTreesClassifier",
        "HistGradientBoostingClassifier",
    )
    return type(estimator).__name__ in tree_types


def _get_transformed_feature_names(preprocessor: Pipeline) -> list[str]:
    """Try to extract feature names from a fitted preprocessing pipeline."""
    try:
        last_step = preprocessor.steps[-1][1]
        if hasattr(last_step, "get_feature_names_out"):
            return list(last_step.get_feature_names_out())
    except Exception:
        pass

    log.debug("_get_transformed_feature_names.fallback")
    return []
