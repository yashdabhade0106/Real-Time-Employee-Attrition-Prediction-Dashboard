"""Admin page — visible to admin role only."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parents[3]
_DB_PATH = _ROOT / "data" / "app.db"


def render() -> None:
    st.markdown("## ⚙️ Admin Panel")
    if st.session_state.get("role") != "admin":
        st.error("🔒 Admin access required.")
        return

    tab1, tab2, tab3 = st.tabs(["Users", "Model Info", "Config"])

    with tab1:
        st.markdown("### User Management")
        try:
            conn = sqlite3.connect(str(_DB_PATH))
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT username, role, created_at FROM users").fetchall()
            conn.close()
            import pandas as pd
            df = pd.DataFrame([dict(r) for r in rows])
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"Could not load users: {e}")

        st.markdown("#### Add User")
        with st.form("admin_add_user"):
            nu = st.text_input("Username")
            np_ = st.text_input("Password", type="password")
            nr = st.selectbox("Role", ["viewer", "hr", "admin"])
            if st.form_submit_button("Add User"):
                import hashlib, sqlite3 as sq
                conn = sq.connect(str(_DB_PATH))
                try:
                    conn.execute(
                        "INSERT INTO users (username, password_hash, role, created_at) VALUES (?,?,?,datetime('now'))",
                        (nu.lower(), hashlib.sha256(np_.encode()).hexdigest(), nr),
                    )
                    conn.commit()
                    st.success(f"User '{nu}' created with role '{nr}'.")
                except Exception as e:
                    st.error(str(e))
                finally:
                    conn.close()

    with tab2:
        st.markdown("### Trained Model Info")
        meta_path = _ROOT / "models" / "v1" / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            st.json(meta)
        else:
            st.warning("No model trained yet. Run `python -m attrition.train`")

    with tab3:
        st.markdown("### App Configuration")
        from attrition.config import get_config
        cfg = get_config()
        st.json({
            "app_env": cfg.app_env,
            "model_version": cfg.model_version,
            "random_seed": cfg.random_seed,
            "prediction_log_retention_days": cfg.prediction_log_retention_days,
            "min_group_size": cfg.min_group_size,
            "batch_scoring_enabled": cfg.batch_scoring_enabled,
        })
