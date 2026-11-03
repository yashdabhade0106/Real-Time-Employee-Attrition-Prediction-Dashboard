"""Model health monitoring page — admin only."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parents[3]


def render() -> None:
    st.markdown("## ❤️ Model Health")
    if st.session_state.get("role") != "admin":
        st.error("🔒 Admin access required.")
        return

    meta_path = _ROOT / "models" / "v1" / "metadata.json"
    if not meta_path.exists():
        st.warning("No model found. Run `python -m attrition.train`.")
        return

    with open(meta_path) as f:
        meta = json.load(f)

    test_m = meta.get("test_metrics", {})
    cv_m = meta.get("cv_metrics", {})

    st.markdown("### Test Set Performance")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ROC-AUC", f"{test_m.get('roc_auc', 0):.4f}")
    c2.metric("Recall", f"{test_m.get('recall', 0):.4f}")
    c3.metric("Precision", f"{test_m.get('precision', 0):.4f}")
    c4.metric("F1", f"{test_m.get('f1', 0):.4f}")
    c5.metric("Brier Score", f"{test_m.get('brier_score', 0):.4f}")

    st.markdown("### Cross-Validation (5-fold × 3 repeats)")
    if cv_m:
        import pandas as pd
        cv_df = pd.DataFrame({
            k: {"Mean": v["mean"], "Std": v["std"]}
            for k, v in cv_m.items()
        }).T.round(4)
        st.dataframe(cv_df, use_container_width=True)

    st.markdown("### Risk Bands")
    bands = meta.get("risk_bands", {})
    for band, info in bands.items():
        if isinstance(info, dict):
            obs = info.get("observed_attrition_rate", "N/A")
            st.markdown(f"**{band.capitalize()}**: observed attrition rate = {obs:.1%}" if isinstance(obs, float) else f"**{band.capitalize()}**: {info}")

    st.markdown("### Model Metadata")
    st.json({k: v for k, v in meta.items() if k not in ("test_metrics", "cv_metrics", "risk_bands")})
