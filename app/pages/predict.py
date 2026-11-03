"""Predict page — single employee attrition prediction form with results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[3]


@st.cache_resource(show_spinner=False)
def _load_model() -> tuple[Any, dict]:
    """Load the trained model pipeline and metadata (cached)."""
    from attrition.predict import load_model
    return load_model(model_dir=str(_ROOT / "models"))


def _model_ready() -> bool:
    return ((_ROOT / "models" / "v1" / "pipeline.joblib").exists() and
            (_ROOT / "models" / "v1" / "metadata.json").exists())


FIELD_GROUPS = {
    "👤 Personal": [
        ("Age", "number", 18, 65, 35, 1, "Employee age in years"),
        ("Gender", "select", ["Male", "Female"], None, None, None, "Gender identity"),
        ("MaritalStatus", "select", ["Single", "Married", "Divorced"], None, None, None, "Marital status"),
        ("DistanceFromHome", "number", 1, 29, 5, 1, "Distance from home (km)"),
        ("Education", "select", [1, 2, 3, 4, 5], None, None, None, "Education level (1=Below College…5=Doctor)"),
        ("EducationField", "select", ["Human Resources", "Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"], None, None, None, "Field of education"),
    ],
    "💼 Job": [
        ("Department", "select", ["Human Resources", "Research & Development", "Sales"], None, None, None, "Department"),
        ("JobRole", "select", ["Healthcare Representative", "Human Resources", "Laboratory Technician", "Manager", "Manufacturing Director", "Research Director", "Research Scientist", "Sales Executive", "Sales Representative"], None, None, None, "Job role"),
        ("JobLevel", "number", 1, 5, 2, 1, "Job level (1=Entry…5=Senior)"),
        ("JobInvolvement", "number", 1, 4, 3, 1, "Job involvement (1=Low…4=Very High)"),
        ("BusinessTravel", "select", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"], None, None, None, "Business travel frequency"),
        ("OverTime", "select", ["No", "Yes"], None, None, None, "Currently working overtime?"),
    ],
    "💰 Compensation": [
        ("MonthlyIncome", "number", 1009, 19999, 5000, 100, "Monthly income (USD)"),
        ("DailyRate", "number", 102, 1499, 700, 10, "Daily rate (USD)"),
        ("HourlyRate", "number", 30, 100, 65, 1, "Hourly rate (USD)"),
        ("MonthlyRate", "number", 2094, 26999, 14000, 100, "Monthly rate (USD)"),
        ("PercentSalaryHike", "number", 11, 25, 14, 1, "Last salary hike %"),
        ("StockOptionLevel", "number", 0, 3, 1, 1, "Stock option level (0=None…3=High)"),
    ],
    "😊 Satisfaction": [
        ("JobSatisfaction", "number", 1, 4, 3, 1, "Job satisfaction (1=Low…4=Very High)"),
        ("EnvironmentSatisfaction", "number", 1, 4, 3, 1, "Environment satisfaction (1=Low…4=Very High)"),
        ("RelationshipSatisfaction", "number", 1, 4, 3, 1, "Relationship satisfaction (1=Low…4=Very High)"),
        ("WorkLifeBalance", "number", 1, 4, 3, 1, "Work-life balance (1=Bad…4=Best)"),
        ("PerformanceRating", "number", 3, 4, 3, 1, "Performance rating (3=Excellent, 4=Outstanding)"),
    ],
    "📅 Experience": [
        ("TotalWorkingYears", "number", 0, 40, 10, 1, "Total working years"),
        ("YearsAtCompany", "number", 0, 40, 5, 1, "Years at current company"),
        ("YearsInCurrentRole", "number", 0, 18, 3, 1, "Years in current role"),
        ("YearsWithCurrManager", "number", 0, 17, 3, 1, "Years with current manager"),
        ("YearsSinceLastPromotion", "number", 0, 15, 1, 1, "Years since last promotion"),
        ("NumCompaniesWorked", "number", 0, 9, 2, 1, "Number of companies worked at"),
        ("TrainingTimesLastYear", "number", 0, 6, 2, 1, "Training sessions last year"),
    ],
}


def _render_form() -> dict[str, Any] | None:
    """Render the grouped employee input form. Returns employee dict or None."""
    employee: dict[str, Any] = {}

    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.markdown("### 📝 Employee Profile Input")

    for group_name, fields in FIELD_GROUPS.items():
        with st.expander(group_name, expanded=(group_name == "👤 Personal")):
            cols = st.columns(3)
            for i, field_def in enumerate(fields):
                name, ftype, *rest = field_def
                with cols[i % 3]:
                    if ftype == "select":
                        options = rest[0]
                        tooltip = rest[4] if len(rest) > 4 else ""
                        employee[name] = st.selectbox(name, options, help=tooltip, key=f"form_{name}")
                    elif ftype == "number":
                        min_v, max_v, default, step = rest[0], rest[1], rest[2], rest[3]
                        tooltip = rest[4] if len(rest) > 4 else ""
                        employee[name] = st.number_input(
                            name, min_value=min_v, max_value=max_v,
                            value=default, step=step, help=tooltip, key=f"form_{name}"
                        )

    st.markdown("---")
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        submitted = st.button("🔍 Predict Attrition", type="primary", use_container_width=True)
    with c2:
        if st.button("🔄 Reset Form", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key.startswith("form_"):
                    del st.session_state[key]
            st.rerun()

    # Cross-field validation
    if submitted:
        errors = []
        if employee.get("YearsAtCompany", 0) > employee.get("TotalWorkingYears", 0):
            errors.append("YearsAtCompany cannot exceed TotalWorkingYears")
        if employee.get("TotalWorkingYears", 0) > employee.get("Age", 100) - 16:
            errors.append("TotalWorkingYears seems too high for the given Age")
        if employee.get("YearsInCurrentRole", 0) > employee.get("YearsAtCompany", 0):
            errors.append("YearsInCurrentRole cannot exceed YearsAtCompany")
        if employee.get("YearsWithCurrManager", 0) > employee.get("YearsAtCompany", 0):
            errors.append("YearsWithCurrManager cannot exceed YearsAtCompany")
        if employee.get("YearsSinceLastPromotion", 0) > employee.get("YearsAtCompany", 0):
            errors.append("YearsSinceLastPromotion cannot exceed YearsAtCompany")
        if errors:
            for e in errors:
                st.error(f"❌ {e}")
            return None
        return employee

    return None


def _render_result(result: dict[str, Any], employee: dict[str, Any]) -> None:
    """Display the prediction result card."""
    prob = result["probability"]
    band = result["risk_band"]
    pred_class = result["class"]
    threshold = result["threshold"]
    version = result.get("model_version", "v1")

    band_color = {"Low": "🟢", "Moderate": "🟡", "High": "🔴"}.get(band, "⚪")
    pred_label = "⚠️ Likely to Leave" if pred_class == 1 else "✅ Likely to Stay"

    st.markdown("---")
    st.markdown("### 🎯 Prediction Result")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Prediction", pred_label)
    col2.metric("Attrition Probability", f"{prob:.1%}")
    col3.metric("Risk Band", f"{band_color} {band}")
    col4.metric("Threshold Used", f"{threshold:.2f}")

    # Colour-coded result card
    bg = {"Low": "#064e3b", "Moderate": "#78350f", "High": "#7f1d1d"}.get(band, "#1e293b")
    border = {"Low": "#10b981", "Moderate": "#f59e0b", "High": "#ef4444"}.get(band, "#6366f1")

    st.markdown(
        f"""
        <div style="background:{bg};border:2px solid {border};border-radius:16px;
                    padding:1.5rem;margin:1rem 0;">
            <h3 style="color:white;margin:0">{pred_label}</h3>
            <p style="color:#e2e8f0;margin:0.3rem 0">
                Calibrated probability: <strong>{prob:.1%}</strong> &nbsp;|&nbsp;
                Risk band: <strong>{band}</strong> &nbsp;|&nbsp;
                Model {version}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="disclaimer">⚠️ This is a statistical estimate from a model trained on fictional IBM HR data. '
        'It must <strong>never</strong> be used as the sole basis for an employment decision.</div>',
        unsafe_allow_html=True,
    )

    # Store in session for insights page
    st.session_state.last_prediction = result
    st.session_state.employee_data = employee

    c1, c2 = st.columns(2)
    with c1:
        if st.button("💡 View Full Insights →", use_container_width=True, type="primary"):
            st.session_state.page = "insights"
            st.rerun()
    with c2:
        if st.button("🔄 Predict Another Employee", use_container_width=True):
            st.session_state.last_prediction = None
            st.session_state.employee_data = {}
            for key in list(st.session_state.keys()):
                if key.startswith("form_"):
                    del st.session_state[key]
            st.rerun()


def render() -> None:
    """Main render function called from app/main.py."""
    st.markdown("## 🔍 Employee Attrition Prediction")

    if not _model_ready():
        st.error("❌ No trained model found. Please run: `python -m attrition.train` in the project root.")
        st.code("python -m attrition.train", language="bash")
        return

    # If we already have a result, show it + option to redo
    if st.session_state.get("last_prediction") and st.session_state.get("employee_data"):
        _render_result(st.session_state.last_prediction, st.session_state.employee_data)
        with st.expander("🔄 Change inputs and repredict"):
            employee = _render_form()
            if employee:
                with st.spinner("Running prediction..."):
                    pipeline, metadata = _load_model()
                    from attrition.predict import predict_single
                    result = predict_single(pipeline, metadata, employee)
                    st.session_state.last_prediction = result
                    st.session_state.employee_data = employee
                    st.rerun()
        return

    employee = _render_form()
    if employee:
        with st.spinner("🔄 Running attrition prediction..."):
            try:
                pipeline, metadata = _load_model()
                from attrition.predict import predict_single
                result = predict_single(pipeline, metadata, employee)
                _render_result(result, employee)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
