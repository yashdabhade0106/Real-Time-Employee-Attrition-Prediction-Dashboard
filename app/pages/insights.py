"""Insights page — SHAP explanations, recommendations, strengths, what-if simulator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[3]


@st.cache_resource(show_spinner=False)
def _load_model() -> tuple[Any, dict]:
    from attrition.predict import load_model
    return load_model(model_dir=str(_ROOT / "models"))


def _get_shap_factors(pipeline: Any, employee: dict[str, Any]) -> list[dict]:
    """Compute SHAP top factors for an employee."""
    try:
        from attrition.explain import compute_shap_values, get_top_factors
        df = pd.DataFrame([employee])
        shap_vals, expected_val, feat_names = compute_shap_values(pipeline, df)
        return get_top_factors(shap_vals[0], feat_names, n_top=10)
    except Exception:
        return []


def render() -> None:
    st.markdown("## 💡 Employee Insights & Recommendations")

    result = st.session_state.get("last_prediction")
    employee = st.session_state.get("employee_data")

    if not result or not employee:
        st.info("👈 First run a prediction on the **Predict** page.")
        if st.button("Go to Predict →", type="primary"):
            st.session_state.page = "predict"
            st.rerun()
        return

    prob = result["probability"]
    band = result["risk_band"]
    pred_class = result["class"]

    # Profile summary
    st.markdown("### 👤 Employee Profile")
    col1, col2, col3 = st.columns(3)
    col1.metric("Age", employee.get("Age", "N/A"))
    col2.metric("Department", employee.get("Department", "N/A"))
    col3.metric("Job Role", employee.get("JobRole", "N/A"))
    col1.metric("Monthly Income", f"${employee.get('MonthlyIncome', 0):,}")
    col2.metric("Years at Company", employee.get("YearsAtCompany", "N/A"))
    col3.metric("Overtime", employee.get("OverTime", "N/A"))

    st.markdown("---")

    # Result summary
    band_emoji = {"Low": "🟢", "Moderate": "🟡", "High": "🔴"}.get(band, "⚪")
    st.markdown(f"### {band_emoji} Risk Assessment: **{band}** ({prob:.1%} probability)")

    # SHAP factors
    st.markdown("### 🔍 Top Factors Influencing This Prediction")
    st.caption("Based on SHAP values — individual feature contributions to the model output.")

    model_ready = ((_ROOT / "models" / "v1" / "pipeline.joblib").exists())
    shap_factors = []

    if model_ready:
        try:
            pipeline, metadata = _load_model()
            shap_factors = _get_shap_factors(pipeline, employee)
        except Exception as e:
            st.warning(f"SHAP computation unavailable: {e}")

    if shap_factors:
        import plotly.graph_objects as go
        features = [f["feature"] for f in shap_factors[:10]]
        values = [f["contribution"] for f in shap_factors[:10]]
        colors = ["#ef4444" if v > 0 else "#10b981" for v in values]
        fig = go.Figure(go.Bar(
            x=values, y=features, orientation="h",
            marker_color=colors,
            text=[f"{v:+.4f}" for v in values],
            textposition="outside",
        ))
        fig.update_layout(
            title="Feature SHAP Contributions (positive = increases attrition risk)",
            xaxis_title="SHAP Value",
            plot_bgcolor="#0f172a",
            paper_bgcolor="#0f172a",
            font_color="#e2e8f0",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("⚠️ SHAP values show model sensitivity, not causal effects.")
    else:
        st.info("SHAP factors not available for this model type.")

    # Recommendations
    st.markdown("---")
    st.markdown("### 📋 Retention Recommendations")
    from attrition.insights import generate_recommendations, get_strengths, get_risk_factors
    recommendations = generate_recommendations(employee, band, shap_factors)

    if recommendations:
        priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        for rec in recommendations[:6]:
            p = rec["priority"]
            with st.expander(f"{priority_icon.get(p,'⚪')} [{p.upper()}] {rec['category']}"):
                st.markdown(f"**{rec['recommendation']}**")
                st.caption(f"_Rationale: {rec['rationale']}_")
    else:
        st.success("✅ No critical risk factors identified.")

    # Strengths & risks
    col1, col2 = st.columns(2)
    strengths = get_strengths(employee, shap_factors)
    risks = get_risk_factors(employee, shap_factors)

    with col1:
        st.markdown("### ✅ Strengths")
        if strengths:
            for s in strengths[:5]:
                st.markdown(f"• {s}")
        else:
            st.markdown("_No specific strengths identified._")

    with col2:
        st.markdown("### ⚠️ Risk Factors")
        if risks:
            for r in risks[:5]:
                st.markdown(f"• {r}")
        else:
            st.markdown("_No specific risk factors identified._")

    # What-if simulator
    st.markdown("---")
    st.markdown("### 🔬 What-If Simulator")
    st.caption("Adjust key factors to see how the model score would change. For illustration only — not a causal model.")

    if model_ready:
        col1, col2 = st.columns(2)
        with col1:
            sim_overtime = st.selectbox("OverTime", ["No", "Yes"],
                index=0 if str(employee.get("OverTime", "No")).lower() != "yes" else 1,
                key="sim_overtime")
            sim_satisfaction = st.slider("JobSatisfaction", 1, 4,
                int(employee.get("JobSatisfaction", 3)), key="sim_sat")
        with col2:
            sim_wlb = st.slider("WorkLifeBalance", 1, 4,
                int(employee.get("WorkLifeBalance", 3)), key="sim_wlb")
            sim_promo = st.number_input("YearsSinceLastPromotion", 0, 15,
                int(employee.get("YearsSinceLastPromotion", 1)), key="sim_promo")

        if st.button("🔄 Run What-If Simulation", use_container_width=True):
            sim_employee = {**employee,
                "OverTime": sim_overtime,
                "JobSatisfaction": sim_satisfaction,
                "WorkLifeBalance": sim_wlb,
                "YearsSinceLastPromotion": sim_promo}
            try:
                pipeline, metadata = _load_model()
                from attrition.predict import predict_single
                sim_result = predict_single(pipeline, metadata, sim_employee)
                delta = sim_result["probability"] - prob
                delta_str = f"+{delta:.1%}" if delta >= 0 else f"{delta:.1%}"
                col1, col2 = st.columns(2)
                col1.metric("Original Probability", f"{prob:.1%}")
                col2.metric("Simulated Probability", f"{sim_result['probability']:.1%}", delta=delta_str)
                st.caption("ℹ️ This is model sensitivity analysis, not a causal prediction.")
            except Exception as exc:
                st.error(f"Simulation failed: {exc}")

    st.markdown(
        '<div class="disclaimer">⚠️ All insights are statistical associations from a fictional dataset. '
        'Consult HR professionals before making any employment decisions.</div>',
        unsafe_allow_html=True,
    )
