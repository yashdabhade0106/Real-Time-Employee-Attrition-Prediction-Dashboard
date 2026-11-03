"""Batch scoring page — CSV upload with validation and downloadable results."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[3]
_MAX_ROWS = 500


@st.cache_resource(show_spinner=False)
def _load_model() -> tuple[Any, dict]:
    from attrition.predict import load_model
    return load_model(model_dir=str(_ROOT / "models"))


REQUIRED_COLUMNS = [
    "Age", "BusinessTravel", "DailyRate", "Department", "DistanceFromHome",
    "Education", "EducationField", "EnvironmentSatisfaction", "Gender",
    "HourlyRate", "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction",
    "MaritalStatus", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
    "OverTime", "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction",
    "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear",
    "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole",
    "YearsSinceLastPromotion", "YearsWithCurrManager",
]


def render() -> None:
    st.markdown("## 📁 Batch Employee Scoring")

    role = st.session_state.get("role", "viewer")
    if role not in ("admin", "hr"):
        st.warning("🔒 Batch scoring is available to HR Analyst and Admin roles only.")
        return

    if not ((_ROOT / "models" / "v1" / "pipeline.joblib").exists()):
        st.error("No trained model found. Run `python -m attrition.train` first.")
        return

    st.markdown(f"Upload a CSV with up to **{_MAX_ROWS} rows** containing employee features.")

    # Template download
    template_df = pd.DataFrame(columns=REQUIRED_COLUMNS)
    csv_template = template_df.to_csv(index=False)
    st.download_button(
        "⬇️ Download CSV Template",
        csv_template.encode(),
        file_name="attrition_batch_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload Employee CSV", type=["csv"], key="batch_upload")
    if not uploaded:
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not parse CSV: {e}")
        return

    if len(df) > _MAX_ROWS:
        st.error(f"File has {len(df)} rows. Maximum allowed: {_MAX_ROWS}.")
        return

    # Column validation
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in REQUIRED_COLUMNS and c != "Attrition"]

    errors = []
    if missing_cols:
        errors.append(f"Missing columns: {', '.join(missing_cols)}")
    if errors:
        for e in errors:
            st.error(f"❌ {e}")
        return

    st.info(f"✅ {len(df)} rows loaded. Running predictions...")

    try:
        pipeline, metadata = _load_model()
        from attrition.predict import predict_batch
        features_df = df[REQUIRED_COLUMNS]
        results_df = predict_batch(pipeline, metadata, features_df)

        # Merge back any extra columns
        for col in extra_cols:
            results_df[col] = df[col].values

        # Summary stats
        n_high = int((results_df["risk_band"] == "High").sum())
        n_mod = int((results_df["risk_band"] == "Moderate").sum())
        n_low = int((results_df["risk_band"] == "Low").sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("🔴 High Risk", n_high)
        c2.metric("🟡 Moderate Risk", n_mod)
        c3.metric("🟢 Low Risk", n_low)

        # Results table
        display_cols = ["attrition_probability", "predicted_class", "risk_band", "model_version"]
        if "JobRole" in results_df.columns:
            display_cols = ["JobRole", "Department"] + display_cols
        st.dataframe(results_df[display_cols].head(100), use_container_width=True)

        # Download
        csv_out = results_df.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download Full Results CSV",
            csv_out,
            file_name="attrition_predictions.csv",
            mime="text/csv",
            type="primary",
        )

        st.markdown(
            '<div class="disclaimer">⚠️ Results are estimates. No raw personal data is stored. '
            'Batch results must not be used for employment decisions without HR review.</div>',
            unsafe_allow_html=True,
        )

    except Exception as exc:
        st.error(f"Batch prediction failed: {exc}")
