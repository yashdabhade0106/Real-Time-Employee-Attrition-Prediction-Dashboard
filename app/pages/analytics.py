"""Team & department analytics page — aggregated attrition insights."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[3]


def render() -> None:
    st.markdown("## 🏢 Team & Department Analytics")
    st.caption("Aggregated views only. Minimum group size of 5 employees to protect privacy.")

    data_path = _ROOT / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    if not data_path.exists():
        st.error("Dataset not found.")
        return

    df = pd.read_csv(data_path)
    df["Attrition_Binary"] = (df["Attrition"] == "Yes").astype(int)

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        dept_filter = st.multiselect(
            "Filter by Department",
            df["Department"].unique().tolist(),
            default=df["Department"].unique().tolist(),
        )
    with col2:
        role_filter = st.multiselect(
            "Filter by Job Role",
            df["JobRole"].unique().tolist(),
            default=df["JobRole"].unique().tolist(),
        )

    filtered = df[df["Department"].isin(dept_filter) & df["JobRole"].isin(role_filter)]

    if len(filtered) < 5:
        st.warning("⚠️ Group size too small (< 5). Aggregates hidden to protect privacy.")
        return

    # KPI row
    total = len(filtered)
    n_attrition = filtered["Attrition_Binary"].sum()
    rate = n_attrition / total

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Employees", total)
    c2.metric("Attrites", n_attrition)
    c3.metric("Attrition Rate", f"{rate:.1%}")
    c4.metric("Avg Monthly Income", f"${filtered['MonthlyIncome'].mean():,.0f}")

    st.markdown("---")

    import plotly.express as px

    # By Department
    dept_agg = (
        filtered.groupby("Department")
        .agg(n=("Attrition_Binary", "count"), attrition_rate=("Attrition_Binary", "mean"))
        .reset_index()
    )
    dept_agg = dept_agg[dept_agg["n"] >= 5]
    dept_agg["attrition_rate_pct"] = (dept_agg["attrition_rate"] * 100).round(1)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Attrition Rate by Department")
        fig = px.bar(
            dept_agg, x="Department", y="attrition_rate_pct",
            color="attrition_rate_pct", color_continuous_scale="RdYlGn_r",
            labels={"attrition_rate_pct": "Attrition %"},
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Attrition Count by Job Role")
        role_agg = (
            filtered.groupby("JobRole")["Attrition_Binary"]
            .agg(["sum", "count", "mean"])
            .rename(columns={"sum": "Attritions", "count": "Total", "mean": "Rate"})
            .reset_index()
        )
        role_agg = role_agg[role_agg["Total"] >= 5]
        role_agg["Rate%"] = (role_agg["Rate"] * 100).round(1)
        fig2 = px.bar(
            role_agg.sort_values("Rate%", ascending=False),
            x="Rate%", y="JobRole", orientation="h",
            color="Rate%", color_continuous_scale="RdYlGn_r",
            template="plotly_dark",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Satisfaction heatmap
    st.markdown("#### Average Satisfaction Scores by Department")
    sat_cols = ["JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance", "RelationshipSatisfaction"]
    sat_agg = filtered.groupby("Department")[sat_cols].mean().round(2)
    st.dataframe(sat_agg.style.background_gradient(cmap="RdYlGn", axis=None, vmin=1, vmax=4),
                 use_container_width=True)

    st.markdown(
        '<div class="disclaimer">Data shown: IBM HR Analytics (fictional). '
        'Groups < 5 are suppressed for privacy.</div>',
        unsafe_allow_html=True,
    )
