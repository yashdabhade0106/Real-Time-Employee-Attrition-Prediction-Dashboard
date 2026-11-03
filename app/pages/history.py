"""Prediction history page."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.markdown("## 📜 Prediction History")
    st.info("Prediction history logging is stored in session only. Enable the database history module for persistent tracking.")
    st.markdown("### Recent Session Predictions")
    result = st.session_state.get("last_prediction")
    employee = st.session_state.get("employee_data")

    if result and employee:
        import pandas as pd
        row = {
            "Timestamp": result.get("timestamp", "N/A"),
            "JobRole": employee.get("JobRole", "N/A"),
            "Department": employee.get("Department", "N/A"),
            "Probability": f"{result.get('probability', 0):.1%}",
            "Risk Band": result.get("risk_band", "N/A"),
            "Prediction": "Leave" if result.get("class") == 1 else "Stay",
            "Model": result.get("model_version", "v1"),
        }
        st.dataframe(pd.DataFrame([row]), use_container_width=True)
    else:
        st.markdown("_No predictions made in this session yet._")
        if st.button("Go to Predict →", type="primary"):
            st.session_state.page = "predict"
            st.rerun()
