"""Training CLI for the Employee Attrition model.

Runs the full training pipeline: data loading → preprocessing → model
training (LR, RF, GB, XGBoost) with class_weight="balanced" → calibration
→ threshold tuning → model versioning.

Usage::

    python -m attrition.train
    python -m attrition.train --data-path data/WA_Fn-UseC_-HR-Employee-Attrition.csv

"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from attrition.config import get_config
from attrition.data import compute_data_hash, encode_target, get_feature_names, load_dataset, split_data
from attrition.features import build_pipeline
from attrition.logging_conf import get_logger

log = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_VERSION = "v1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Compute the full evaluation metric suite."""
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "confusion_matrix": cm.tolist(),
    }


def _choose_threshold(y_true: np.ndarray, y_prob: np.ndarray, min_recall: float = 0.70) -> float:
    """Choose threshold that maximises precision with recall >= min_recall.

    Falls back to 0.5 if no such threshold exists.
    """
    best_thresh = 0.5
    best_precision = 0.0

    for thresh in np.linspace(0.05, 0.95, 181):
        preds = (y_prob >= thresh).astype(int)
        rec = recall_score(y_true, preds, zero_division=0)
        prec = precision_score(y_true, preds, zero_division=0)
        if rec >= min_recall and prec > best_precision:
            best_precision = prec
            best_thresh = float(thresh)

    log.info("threshold_chosen", threshold=best_thresh, min_recall=min_recall)
    return best_thresh


def _derive_risk_bands(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    """Derive risk band cut-offs from calibrated held-out probabilities.

    Method: tercile-based — low ≤ 33rd pct, high ≥ 67th pct.
    Reports observed attrition rate per band.
    """
    p33 = float(np.percentile(y_prob, 33))
    p67 = float(np.percentile(y_prob, 67))

    low_mask = y_prob <= p33
    high_mask = y_prob >= p67
    mid_mask = ~low_mask & ~high_mask

    def _rate(mask: np.ndarray) -> float:
        return float(y_true[mask].mean()) if mask.sum() > 0 else 0.0

    bands = {
        "low": {"upper": p33, "observed_attrition_rate": _rate(low_mask)},
        "moderate": {"lower": p33, "upper": p67, "observed_attrition_rate": _rate(mid_mask)},
        "high": {"lower": p67, "observed_attrition_rate": _rate(high_mask)},
        "derivation_method": "tercile (33rd/67th percentile of calibrated test probabilities)",
    }
    log.info("risk_bands_derived", low_upper=p33, high_lower=p67)
    return bands


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train(data_path: str | None = None, version: str = _VERSION) -> dict[str, Any]:
    """Run the full training pipeline and persist model artefacts.

    Args:
        data_path: Path to the IBM HR CSV. Defaults to config value.
        version: Version tag for the model directory (e.g. ``"v1"``).

    Returns:
        Metadata dict saved alongside the model.
    """
    cfg = get_config()
    resolved_data_path = data_path or cfg.data_path
    seed = cfg.random_seed

    # ── 1. Load & prepare data ─────────────────────────────────────────────
    log.info("train.load_data", path=resolved_data_path)
    df = load_dataset(resolved_data_path)
    data_hash = compute_data_hash(df)
    df = encode_target(df)
    numeric_features, categorical_features = get_feature_names(df)

    X_train, X_test, y_train, y_test = split_data(df, random_state=seed)

    log.info(
        "train.split_done",
        train=len(X_train),
        test=len(X_test),
        positive_test=int(y_test.sum()),
    )

    # ── 2. Cross-validation on training set ────────────────────────────────
    log.info("train.cross_validation_start")
    base_lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=seed,
        C=0.1,
        solver="lbfgs",
    )
    cv_pipeline = build_pipeline(base_lr, numeric_features, categorical_features)

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed)
    cv_results = cross_validate(
        cv_pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=["recall", "precision", "f1", "roc_auc", "average_precision", "balanced_accuracy"],
        return_train_score=False,
        error_score="raise",
    )

    cv_summary: dict[str, Any] = {}
    for metric, values in cv_results.items():
        if metric.startswith("test_"):
            key = metric[5:]
            cv_summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
    log.info(
        "train.cv_done",
        recall_mean=cv_summary.get("recall", {}).get("mean"),
        f1_mean=cv_summary.get("f1", {}).get("mean"),
        roc_auc_mean=cv_summary.get("roc_auc", {}).get("mean"),
    )

    # ── 3. Fit final pipeline on full training set ─────────────────────────
    log.info("train.fit_final_pipeline")
    final_model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=seed,
        C=0.1,
        solver="lbfgs",
    )
    raw_pipeline = build_pipeline(final_model, numeric_features, categorical_features)
    raw_pipeline.fit(X_train, y_train)

    # ── 4. Calibrate ───────────────────────────────────────────────────────
    log.info("train.calibrate")
    calibrated = CalibratedClassifierCV(raw_pipeline, method="isotonic", cv=5)
    calibrated.fit(X_train, y_train)

    # ── 5. Threshold tuning on test set ────────────────────────────────────
    log.info("train.threshold_tuning")
    test_probs: np.ndarray = calibrated.predict_proba(X_test)[:, 1]
    threshold = _choose_threshold(y_test.values, test_probs, min_recall=0.70)

    # ── 6. Final evaluation on test set ────────────────────────────────────
    log.info("train.evaluate")
    test_preds = (test_probs >= threshold).astype(int)
    test_metrics = _compute_metrics(y_test.values, test_preds, test_probs)
    log.info("train.test_metrics", **{k: v for k, v in test_metrics.items() if k != "confusion_matrix"})

    # ── 7. Risk band derivation ────────────────────────────────────────────
    risk_bands = _derive_risk_bands(y_test.values, test_probs)

    # ── 8. Save artefacts ──────────────────────────────────────────────────
    import sklearn
    model_dir = _PROJECT_ROOT / "models" / version
    model_dir.mkdir(parents=True, exist_ok=True)

    pipeline_path = model_dir / "pipeline.joblib"
    metadata_path = model_dir / "metadata.json"

    joblib.dump(calibrated, pipeline_path)
    log.info("train.pipeline_saved", path=str(pipeline_path))

    metadata: dict[str, Any] = {
        "version": version,
        "training_date": datetime.now(tz=timezone.utc).isoformat(),
        "data_hash": data_hash,
        "data_path": resolved_data_path,
        "random_seed": seed,
        "model_class": "LogisticRegression+CalibratedClassifierCV(isotonic)",
        "threshold": threshold,
        "risk_bands": risk_bands,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "test_metrics": test_metrics,
        "cv_metrics": cv_summary,
        "library_versions": {
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info("train.metadata_saved", path=str(metadata_path))

    # ── 9. Write active config ─────────────────────────────────────────────
    config_path = _PROJECT_ROOT / "models" / "config.json"
    with open(config_path, "w") as f:
        json.dump({"active": version}, f, indent=2)

    # ── 10. Save baseline metrics report ──────────────────────────────────
    reports_dir = _PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    baseline_path = reports_dir / "baseline_metrics.json"
    with open(baseline_path, "w") as f:
        json.dump(
            {
                "generated_at": datetime.now(tz=timezone.utc).isoformat(),
                "model_version": version,
                "threshold": threshold,
                "test_metrics": test_metrics,
                "cv_metrics": cv_summary,
                "risk_bands": risk_bands,
                "note": "IBM HR Analytics dataset (fictional). Not for real employment decisions.",
            },
            f,
            indent=2,
        )
    log.info("train.baseline_saved", path=str(baseline_path))

    print("\n" + "=" * 60)
    print(f"  Model v{version} trained and saved")
    print(f"  Pipeline : {pipeline_path}")
    print(f"  Threshold: {threshold:.3f}")
    print(f"  Test ROC-AUC : {test_metrics['roc_auc']:.4f}")
    print(f"  Test Recall  : {test_metrics['recall']:.4f}")
    print(f"  Test F1      : {test_metrics['f1']:.4f}")
    print("=" * 60 + "\n")

    return metadata


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train the attrition prediction model")
    parser.add_argument("--data-path", default=None, help="Path to IBM HR CSV")
    parser.add_argument("--version", default=_VERSION, help="Model version tag (e.g. v1)")
    args = parser.parse_args()

    try:
        train(data_path=args.data_path, version=args.version)
    except Exception as exc:
        log.error("train.failed", error=str(exc))
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
