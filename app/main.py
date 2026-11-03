"""
RetentionAI — Real-Time Employee Attrition Dashboard
Single-file Streamlit app. All pages are rendered from this one file.
Run: streamlit run app/main.py
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Path setup (must happen before any project imports) ───────────────────────
_APP_DIR = Path(__file__).resolve().parent
_ROOT    = _APP_DIR.parent
_SRC_DIR = _ROOT / "src"
for _p in [str(_ROOT), str(_SRC_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RetentionAI — Attrition Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif !important;
  }
  h1,h2,h3,h4 { font-family: 'Space Grotesk', sans-serif !important; }

  /* ─── App background ─── */
  .stApp {
    background: linear-gradient(135deg, #0a0e27 0%, #0d1b3e 40%, #0f2460 100%);
    min-height: 100vh;
  }

  /* ─── Sidebar ─── */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060d1f 0%, #0d1b3e 100%) !important;
    border-right: 1px solid rgba(99,102,241,0.3);
  }
  section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
  section[data-testid="stSidebar"] .stButton > button {
    background: rgba(99,102,241,0.12) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 10px !important;
    color: #c7d2fe !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    margin-bottom: 4px !important;
  }
  section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(99,102,241,0.35) !important;
    border-color: #6366f1 !important;
    color: #fff !important;
    transform: translateX(3px);
  }

  /* ─── Main content text ─── */
  .main .block-container { padding-top: 2rem; }
  p, li, label, span { color: #dde4f5 !important; }

  /* ─── Gradient page heading ─── */
  .page-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #818cf8, #a78bfa, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
    line-height: 1.2;
  }
  .page-subtitle {
    font-size: 1.05rem;
    color: #94a3b8 !important;
    margin-bottom: 1.5rem;
  }

  /* ─── Glass card ─── */
  .glass-card {
    background: rgba(15, 36, 96, 0.55);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 18px;
    padding: 1.8rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    margin-bottom: 1.2rem;
  }

  /* ─── Metric cards ─── */
  [data-testid="stMetricValue"] { color: #818cf8 !important; font-weight: 700 !important; font-size: 1.5rem !important; }
  [data-testid="stMetricLabel"] { color: #94a3b8 !important; }

  /* ─── Risk badges ─── */
  .risk-high   { display:inline-block; background:#7f1d1d; color:#fca5a5 !important; border:1px solid #ef4444; border-radius:30px; padding:5px 18px; font-weight:700; font-size:1rem; }
  .risk-medium { display:inline-block; background:#78350f; color:#fcd34d !important; border:1px solid #f59e0b; border-radius:30px; padding:5px 18px; font-weight:700; font-size:1rem; }
  .risk-low    { display:inline-block; background:#064e3b; color:#6ee7b7 !important; border:1px solid #10b981; border-radius:30px; padding:5px 18px; font-weight:700; font-size:1rem; }

  /* ─── Result card ─── */
  .result-high   { background:linear-gradient(135deg,#450a0a,#7f1d1d); border:2px solid #ef4444; border-radius:18px; padding:2rem; }
  .result-medium { background:linear-gradient(135deg,#431407,#78350f); border:2px solid #f59e0b; border-radius:18px; padding:2rem; }
  .result-low    { background:linear-gradient(135deg,#022c22,#064e3b); border:2px solid #10b981; border-radius:18px; padding:2rem; }
  .result-high h3, .result-medium h3, .result-low h3 { color: #fff !important; margin:0; }
  .result-high p,  .result-medium p,  .result-low p  { color: #e2e8f0 !important; }

  /* ─── Form inputs ─── */
  .stTextInput input, .stNumberInput input, .stSelectbox select {
    background: rgba(15,36,96,0.6) !important;
    border: 1px solid rgba(99,102,241,0.4) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
  }
  .stTabs [data-baseweb="tab"] { color: #94a3b8 !important; }
  .stTabs [aria-selected="true"] { color: #818cf8 !important; border-bottom: 2px solid #818cf8 !important; }
  .stExpander { background: rgba(15,36,96,0.4) !important; border: 1px solid rgba(99,102,241,0.25) !important; border-radius:12px !important; }
  .stExpander summary { color: #c7d2fe !important; }

  /* ─── Buttons ─── */
  .stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-family: 'Poppins', sans-serif !important;
    transition: all 0.25s ease !important;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #6366f1, #9333ea) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99,102,241,0.5) !important;
  }

  /* ─── Login card ─── */
  .login-card {
    background: rgba(15,36,96,0.7);
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 24px;
    padding: 2.5rem 2rem;
    backdrop-filter: blur(20px);
    box-shadow: 0 20px 60px rgba(0,0,0,0.5);
  }

  /* ─── Disclaimer ─── */
  .disclaimer {
    background: rgba(120,53,15,0.25);
    border: 1px solid rgba(245,158,11,0.35);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    font-size: 0.78rem;
    color: #fbbf24 !important;
    margin-top: 1rem;
  }

  /* ─── Info / warning overrides ─── */
  .stAlert { border-radius: 12px !important; }

  /* ─── Sidebar brand ─── */
  .brand-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    background: linear-gradient(90deg, #818cf8, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
_DB_PATH = _ROOT / "data" / "app.db"
_MODEL_DIR = _ROOT / "models" / "v1"
_DATA_PATH = _ROOT / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  DATABASE / AUTH HELPERS                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _ensure_db() -> None:
    _DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'viewer',
            created_at TEXT NOT NULL
        )
    """)
    cur = conn.execute("SELECT COUNT(*) FROM users")
    if cur.fetchone()[0] == 0:
        conn.execute(
            "INSERT INTO users (username,password_hash,role,created_at) VALUES (?,?,?,?)",
            ("admin", _hash_pw("admin123"), "admin", datetime.now(timezone.utc).isoformat()),
        )
        conn.execute(
            "INSERT INTO users (username,password_hash,role,created_at) VALUES (?,?,?,?)",
            ("hr", _hash_pw("hr1234"), "hr", datetime.now(timezone.utc).isoformat()),
        )
    conn.commit(); conn.close()

def _hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def _authenticate(user: str, pw: str) -> dict | None:
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM users WHERE username=?", (user.strip().lower(),)).fetchone()
    conn.close()
    if row and row["password_hash"] == _hash_pw(pw):
        return dict(row)
    return None

def _register(user: str, pw: str) -> bool:
    try:
        conn = sqlite3.connect(str(_DB_PATH))
        conn.execute(
            "INSERT INTO users (username,password_hash,role,created_at) VALUES (?,?,?,?)",
            (user.strip().lower(), _hash_pw(pw), "viewer", datetime.now(timezone.utc).isoformat()),
        )
        conn.commit(); conn.close(); return True
    except sqlite3.IntegrityError:
        return False

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODEL HELPERS                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@st.cache_resource(show_spinner="Loading AI model…")
def _load_model() -> tuple[Any, dict]:
    from attrition.predict import load_model
    return load_model(model_dir=str(_ROOT / "models"))

def _model_ready() -> bool:
    return (_MODEL_DIR / "pipeline.joblib").exists() and (_MODEL_DIR / "metadata.json").exists()

def _load_meta() -> dict:
    if (_MODEL_DIR / "metadata.json").exists():
        with open(_MODEL_DIR / "metadata.json") as f:
            return json.load(f)
    return {}

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SESSION INIT                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _init_session() -> None:
    defaults = {
        "authenticated": False, "user": None, "role": "viewer",
        "page": "home", "last_prediction": None, "employee_data": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  LOGIN PAGE                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _page_login() -> None:
    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        st.markdown("""
        <div style="text-align:center; padding: 2rem 0 1.5rem 0;">
          <div class="page-title" style="font-size:3rem; text-align:center;">🎯 RetentionAI</div>
          <div class="page-subtitle" style="text-align:center; font-size:1.1rem;">
            Real-Time Employee Attrition Prediction
          </div>
        </div>
        """, unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["🔐 Sign In", "📝 Register"])

        with tab1:
            with st.form("login"):
                st.markdown("#### Welcome Back")
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter password")
                ok = st.form_submit_button("🚀 Sign In", use_container_width=True, type="primary")
            if ok:
                u = _authenticate(username, password)
                if u:
                    st.session_state.update(authenticated=True, user=u["username"], role=u["role"])
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Try admin / admin123")

        with tab2:
            with st.form("register"):
                st.markdown("#### Create Account")
                nu = st.text_input("Username", key="r_u", placeholder="Choose a username")
                np1 = st.text_input("Password", type="password", key="r_p1")
                np2 = st.text_input("Confirm Password", type="password", key="r_p2")
                rok = st.form_submit_button("✅ Create Account", use_container_width=True)
            if rok:
                if np1 != np2:     st.error("Passwords don't match")
                elif len(np1) < 6: st.error("Password too short (min 6)")
                elif _register(nu, np1): st.success("Account created! Sign in above.")
                else:              st.error("Username already taken")

        st.markdown("""
        <div class="disclaimer">
          ⚠️ Demo accounts: <b>admin / admin123</b> (Admin) | <b>hr / hr1234</b> (HR Analyst).<br>
          This system uses a <b>fictional IBM HR dataset</b>. Predictions are statistical estimates 
          and must <b>never</b> be the sole basis for employment decisions.
        </div>
        """, unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SIDEBAR                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _sidebar() -> None:
    role = st.session_state.role
    with st.sidebar:
        st.markdown('<div class="brand-title">🎯 RetentionAI</div>', unsafe_allow_html=True)
        st.markdown(f"<small style='color:#64748b'>Signed in as <b style='color:#818cf8'>{st.session_state.user}</b> ({role})</small>", unsafe_allow_html=True)
        st.markdown("---")

        nav = [
            ("home",      "🏠  Dashboard"),
            ("predict",   "🔍  Predict Attrition"),
            ("insights",  "💡  Insights & SHAP"),
            ("analytics", "🏢  Team Analytics"),
            ("batch",     "📁  Batch Scoring"),
            ("history",   "📜  History"),
        ]
        if role == "admin":
            nav += [("admin","⚙️  Admin Panel"), ("health","❤️  Model Health")]

        for key, label in nav:
            active = st.session_state.page == key
            btn_style = "primary" if active else "secondary"
            if st.button(label, key=f"nav_{key}", use_container_width=True):
                st.session_state.page = key
                st.rerun()

        st.markdown("---")
        if st.button("🚪  Sign Out", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  HOME PAGE                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _page_home() -> None:
    st.markdown('<div class="page-title">🎯 RetentionAI Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Real-Time Employee Attrition Prediction · IBM HR Analytics Dataset</div>', unsafe_allow_html=True)

    # Quick-nav cards
    cols = st.columns(4)
    cards = [
        ("🔍 Predict", "Single-employee attrition prediction with AI insights", "predict", "#4f46e5"),
        ("💡 Insights", "SHAP explanations & retention recommendations", "insights", "#7c3aed"),
        ("🏢 Analytics", "Department & team-level aggregated views", "analytics", "#0891b2"),
        ("📁 Batch", "Upload CSV for bulk team scoring", "batch", "#059669"),
    ]
    for col, (title, desc, pg, color) in zip(cols, cards):
        with col:
            st.markdown(f"""
            <div class="glass-card" style="border-top: 3px solid {color}; min-height:130px;">
              <b style="color:{color}; font-size:1.05rem">{title}</b><br>
              <span style="color:#94a3b8; font-size:0.85rem">{desc}</span>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Open →", key=f"h_{pg}", use_container_width=True):
                st.session_state.page = pg; st.rerun()

    # Model performance
    st.markdown("### 📊 Model Performance")
    meta = _load_meta()
    if meta:
        m = meta.get("test_metrics", {})
        cv = meta.get("cv_metrics", {})
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("ROC-AUC",  f"{m.get('roc_auc',0):.4f}")
        c2.metric("Recall",   f"{m.get('recall',0):.4f}")
        c3.metric("Precision",f"{m.get('precision',0):.4f}")
        c4.metric("F1-Score", f"{m.get('f1',0):.4f}")
        c5.metric("Brier",    f"{m.get('brier_score',0):.4f}")

        cv_recall = cv.get("recall", {})
        if cv_recall:
            st.info(f"📈 5-Fold CV Recall: **{cv_recall.get('mean',0):.4f} ± {cv_recall.get('std',0):.4f}** | "
                    f"Model: {meta.get('model_class','LR')} | Trained: {meta.get('training_date','N/A')[:10]} | "
                    f"Threshold: {meta.get('threshold',0.5):.3f}")
    else:
        st.warning("⚠️ No model found. Run: `python -m attrition.train` from the project root.")
        st.code("cd \"d:/download/ADS LAB/ADS Project/final project/Real-Time-Employee-Attrition-Prediction-Dashboard\"\n$env:PYTHONPATH='src'; python -m attrition.train", language="powershell")

    st.markdown('<div class="disclaimer">⚠️ IBM HR Analytics dataset is fictional (1,470 synthetic records). All predictions are statistical estimates — not employment decisions.</div>', unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PREDICT PAGE                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

FIELD_GROUPS: dict[str, list[tuple]] = {
    "👤 Personal": [
        ("Age","num",18,65,35,1,"Age in years"),
        ("Gender","sel",["Male","Female"],None,None,None,"Gender"),
        ("MaritalStatus","sel",["Single","Married","Divorced"],None,None,None,"Marital status"),
        ("DistanceFromHome","num",1,29,9,1,"Distance from home (km)"),
        ("Education","num",1,5,3,1,"Education level (1=Below College … 5=Doctor)"),
        ("EducationField","sel",["Human Resources","Life Sciences","Marketing","Medical","Other","Technical Degree"],None,None,None,"Education field"),
    ],
    "💼 Job": [
        ("Department","sel",["Human Resources","Research & Development","Sales"],None,None,None,"Department"),
        ("JobRole","sel",["Healthcare Representative","Human Resources","Laboratory Technician","Manager","Manufacturing Director","Research Director","Research Scientist","Sales Executive","Sales Representative"],None,None,None,"Job role"),
        ("JobLevel","num",1,5,2,1,"Job level (1=Entry … 5=Senior)"),
        ("JobInvolvement","num",1,4,3,1,"Job involvement (1=Low … 4=Very High)"),
        ("BusinessTravel","sel",["Non-Travel","Travel_Rarely","Travel_Frequently"],None,None,None,"Travel frequency"),
        ("OverTime","sel",["No","Yes"],None,None,None,"Currently on overtime?"),
    ],
    "💰 Compensation": [
        ("MonthlyIncome","num",1009,19999,5000,100,"Monthly income (USD)"),
        ("DailyRate","num",102,1499,800,10,"Daily rate"),
        ("HourlyRate","num",30,100,66,1,"Hourly rate"),
        ("MonthlyRate","num",2094,26999,14000,100,"Monthly rate"),
        ("PercentSalaryHike","num",11,25,14,1,"Last salary hike %"),
        ("StockOptionLevel","num",0,3,1,1,"Stock option level (0=None … 3=High)"),
    ],
    "😊 Satisfaction": [
        ("JobSatisfaction","num",1,4,3,1,"1=Low … 4=Very High"),
        ("EnvironmentSatisfaction","num",1,4,3,1,"1=Low … 4=Very High"),
        ("RelationshipSatisfaction","num",1,4,3,1,"1=Low … 4=Very High"),
        ("WorkLifeBalance","num",1,4,3,1,"1=Bad … 4=Best"),
        ("PerformanceRating","num",3,4,3,1,"3=Excellent, 4=Outstanding"),
    ],
    "📅 Experience": [
        ("TotalWorkingYears","num",0,40,10,1,"Total working years (all companies)"),
        ("YearsAtCompany","num",0,40,5,1,"Years at current company"),
        ("YearsInCurrentRole","num",0,18,3,1,"Years in current role"),
        ("YearsWithCurrManager","num",0,17,3,1,"Years with current manager"),
        ("YearsSinceLastPromotion","num",0,15,1,1,"Years since last promotion"),
        ("NumCompaniesWorked","num",0,9,2,1,"Number of prior companies"),
        ("TrainingTimesLastYear","num",0,6,2,1,"Training sessions last year"),
    ],
}

def _render_form() -> dict | None:
    emp: dict[str, Any] = {}
    for group, fields in FIELD_GROUPS.items():
        with st.expander(group, expanded=(group == "👤 Personal")):
            cols = st.columns(3)
            for i, fd in enumerate(fields):
                name, ftype = fd[0], fd[1]
                with cols[i % 3]:
                    if ftype == "sel":
                        options = fd[2]
                        emp[name] = st.selectbox(name, options, key=f"f_{name}", help=str(fd[6]))
                    else:
                        mn, mx, dv, st_ = int(fd[2]), int(fd[3]), int(fd[4]), int(fd[5])
                        emp[name] = st.number_input(name, min_value=mn, max_value=mx, value=dv, step=st_, key=f"f_{name}", help=str(fd[6]))

    st.markdown("---")
    c1, c2 = st.columns([1, 1])
    with c1:
        submitted = st.button("🔍 Predict Attrition", type="primary", use_container_width=True)
    with c2:
        if st.button("🔄 Reset All Fields", use_container_width=True):
            for k in [k for k in st.session_state if k.startswith("f_")]:
                del st.session_state[k]
            st.session_state.last_prediction = None
            st.session_state.employee_data = {}
            st.rerun()

    if not submitted:
        return None

    errors = []
    if emp.get("YearsAtCompany", 0) > emp.get("TotalWorkingYears", 0):
        errors.append("YearsAtCompany cannot exceed TotalWorkingYears")
    if emp.get("TotalWorkingYears", 0) > emp.get("Age", 100) - 16:
        errors.append("TotalWorkingYears is too high for the given Age")
    if emp.get("YearsInCurrentRole", 0) > emp.get("YearsAtCompany", 0):
        errors.append("YearsInCurrentRole cannot exceed YearsAtCompany")
    if emp.get("YearsWithCurrManager", 0) > emp.get("YearsAtCompany", 0):
        errors.append("YearsWithCurrManager cannot exceed YearsAtCompany")
    if emp.get("YearsSinceLastPromotion", 0) > emp.get("YearsAtCompany", 0):
        errors.append("YearsSinceLastPromotion cannot exceed YearsAtCompany")
    for e in errors:
        st.error(f"❌ {e}")
    return emp if not errors else None


def _show_result(result: dict, emp: dict) -> None:
    prob  = result["probability"]
    band  = result["risk_band"]
    pred  = result["class"]
    thr   = result["threshold"]
    ver   = result.get("model_version","v1")

    css_cls = {"High":"result-high","Moderate":"result-medium","Low":"result-low"}.get(band,"result-low")
    badge   = {"High":"risk-high","Moderate":"risk-medium","Low":"risk-low"}.get(band,"risk-low")
    label   = "⚠️ Likely to Leave" if pred == 1 else "✅ Likely to Stay"
    emoji   = {"High":"🔴","Moderate":"🟡","Low":"🟢"}.get(band,"⚪")

    st.markdown(f"""
    <div class="{css_cls}">
      <h3>{label}</h3>
      <p>Calibrated probability: <strong>{prob:.1%}</strong> &nbsp;|&nbsp;
         Risk Band: <span class="{badge}">{emoji} {band}</span> &nbsp;|&nbsp;
         Threshold: {thr:.3f} &nbsp;|&nbsp; Model {ver}</p>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Attrition Risk", f"{prob:.1%}")
    c2.metric("Risk Band", f"{emoji} {band}")
    c3.metric("Decision Threshold", f"{thr:.3f}")
    c4.metric("Model Version", ver)

    st.session_state.last_prediction = result
    st.session_state.employee_data   = emp

    st.markdown('<div class="disclaimer">⚠️ Statistical estimate only. Never use as the sole basis for any employment decision.</div>', unsafe_allow_html=True)

    ca, cb = st.columns(2)
    with ca:
        if st.button("💡 View Full AI Insights →", type="primary", use_container_width=True):
            st.session_state.page = "insights"; st.rerun()
    with cb:
        if st.button("🔄 Predict Another Employee", use_container_width=True):
            for k in [k for k in st.session_state if k.startswith("f_")]:
                del st.session_state[k]
            st.session_state.last_prediction = None
            st.session_state.employee_data = {}
            st.rerun()


def _page_predict() -> None:
    st.markdown('<div class="page-title">🔍 Predict Employee Attrition</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Fill in the employee profile below and click Predict</div>', unsafe_allow_html=True)

    if not _model_ready():
        st.error("❌ No trained model found.")
        st.code("$env:PYTHONPATH='src'; python -m attrition.train", language="powershell")
        return

    if st.session_state.get("last_prediction") and st.session_state.get("employee_data"):
        _show_result(st.session_state.last_prediction, st.session_state.employee_data)
        with st.expander("🔄 Run a new prediction"):
            emp = _render_form()
            if emp:
                with st.spinner("Running prediction…"):
                    try:
                        pipeline, meta = _load_model()
                        from attrition.predict import predict_single
                        result = predict_single(pipeline, meta, emp)
                        _show_result(result, emp)
                        st.rerun()
                    except Exception as ex:
                        st.error(f"Prediction error: {ex}")
        return

    emp = _render_form()
    if emp:
        with st.spinner("🤖 Running AI prediction…"):
            try:
                pipeline, meta = _load_model()
                from attrition.predict import predict_single
                result = predict_single(pipeline, meta, emp)
                _show_result(result, emp)
            except Exception as ex:
                st.error(f"Prediction failed: {ex}")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  INSIGHTS PAGE                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _page_insights() -> None:
    st.markdown('<div class="page-title">💡 Employee Insights & AI Explanations</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">SHAP-based feature contributions · Retention recommendations · What-if simulator</div>', unsafe_allow_html=True)

    result = st.session_state.get("last_prediction")
    emp    = st.session_state.get("employee_data")

    if not result or not emp:
        st.info("👈 Run a prediction on the **Predict** page first, then come back here.")
        if st.button("Go to Predict →", type="primary"):
            st.session_state.page = "predict"; st.rerun()
        return

    prob = result["probability"]
    band = result["risk_band"]
    emoji = {"High":"🔴","Moderate":"🟡","Low":"🟢"}.get(band,"⚪")

    # Profile summary
    st.markdown("### 👤 Employee Snapshot")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Age", emp.get("Age","—"))
    c2.metric("Department", emp.get("Department","—"))
    c3.metric("Job Level", emp.get("JobLevel","—"))
    c4.metric("Monthly Income", f"${emp.get('MonthlyIncome',0):,}")
    c5.metric("Overtime", emp.get("OverTime","—"))

    badge = {"High":"risk-high","Moderate":"risk-medium","Low":"risk-low"}.get(band,"risk-low")
    st.markdown(f"""
    <div class="glass-card" style="text-align:center; padding:1rem;">
      <div style="font-size:1.2rem; color:#94a3b8">Attrition Risk Score</div>
      <div style="font-size:3rem; font-weight:800; color:#818cf8">{prob:.1%}</div>
      <span class="{badge}">{emoji} {band} Risk</span>
    </div>
    """, unsafe_allow_html=True)

    # SHAP chart
    st.markdown("### 🔍 Top Contributing Factors (SHAP)")
    st.caption("Positive values push attrition risk UP · Negative values push it DOWN")

    shap_factors: list[dict] = []
    if _model_ready():
        try:
            import pandas as pd
            pipeline, meta = _load_model()
            from attrition.explain import compute_shap_values, get_top_factors
            df = pd.DataFrame([emp])
            sv, _, fn = compute_shap_values(pipeline, df)
            shap_factors = get_top_factors(sv[0], fn, n_top=12)
        except Exception as ex:
            st.warning(f"SHAP unavailable for this model type: {ex}")

    if shap_factors:
        import plotly.graph_objects as go
        feats  = [f["feature"].replace("num__","").replace("cat__","") for f in shap_factors]
        vals   = [f["contribution"] for f in shap_factors]
        colors = ["#ef4444" if v > 0 else "#10b981" for v in vals]
        fig = go.Figure(go.Bar(
            x=vals, y=feats, orientation="h",
            marker_color=colors,
            text=[f"{v:+.4f}" for v in vals], textposition="outside",
        ))
        fig.update_layout(
            xaxis_title="SHAP Value (contribution to log-odds)",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0", family="Poppins"),
            height=420, margin=dict(l=10,r=30,t=10,b=10),
            xaxis=dict(gridcolor="rgba(99,102,241,0.15)", zerolinecolor="#6366f1"),
            yaxis=dict(gridcolor="rgba(99,102,241,0.15)"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("⚠️ SHAP values show model sensitivity — not causal effects. Correlation ≠ Causation.")
    else:
        st.info("SHAP chart requires the calibrated pipeline to support LinearExplainer.")

    # Recommendations
    st.markdown("---")
    st.markdown("### 📋 Retention Recommendations")
    try:
        from attrition.insights import generate_recommendations, get_strengths, get_risk_factors
        recs     = generate_recommendations(emp, band, shap_factors)
        strengths = get_strengths(emp, shap_factors)
        risks     = get_risk_factors(emp, shap_factors)
    except Exception:
        recs = strengths = risks = []

    if recs:
        p_icon = {"high":"🔴","medium":"🟡","low":"🟢"}
        for r in recs[:6]:
            p = r["priority"]
            with st.expander(f"{p_icon.get(p,'⚪')} [{p.upper()}] {r['category']}"):
                st.markdown(f"**{r['recommendation']}**")
                st.caption(f"_Rationale: {r['rationale']}_")
    else:
        st.success("✅ No high-priority risk factors detected.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### ✅ Strengths")
        for s in (strengths or ["_None identified_"])[:5]:
            st.markdown(f"• {s}")
    with col2:
        st.markdown("#### ⚠️ Risk Flags")
        for r in (risks or ["_None identified_"])[:5]:
            st.markdown(f"• {r}")

    # What-if simulator
    st.markdown("---")
    st.markdown("### 🔬 What-If Simulator")
    st.caption("Adjust key levers and see how the model score changes. Model sensitivity only — not causal.")
    if _model_ready():
        wc1, wc2 = st.columns(2)
        with wc1:
            w_ot  = st.selectbox("OverTime", ["No","Yes"], index=0 if emp.get("OverTime","No")!="Yes" else 1, key="w_ot")
            w_js  = st.slider("JobSatisfaction", 1, 4, int(emp.get("JobSatisfaction",3)), key="w_js")
        with wc2:
            w_wlb = st.slider("WorkLifeBalance", 1, 4, int(emp.get("WorkLifeBalance",3)), key="w_wlb")
            w_pr  = st.number_input("YearsSinceLastPromotion", 0, 15, int(emp.get("YearsSinceLastPromotion",1)), key="w_pr")

        if st.button("▶️ Run Simulation", use_container_width=True):
            sim = {**emp, "OverTime":w_ot, "JobSatisfaction":w_js, "WorkLifeBalance":w_wlb, "YearsSinceLastPromotion":int(w_pr)}
            try:
                pipeline, meta = _load_model()
                from attrition.predict import predict_single
                sim_r = predict_single(pipeline, meta, sim)
                delta = sim_r["probability"] - prob
                s1, s2, s3 = st.columns(3)
                s1.metric("Original Probability", f"{prob:.1%}")
                s2.metric("Simulated Probability", f"{sim_r['probability']:.1%}", delta=f"{delta:+.1%}")
                s3.metric("Simulated Risk Band", sim_r["risk_band"])
                st.caption("ℹ️ Model sensitivity analysis — not a causal prediction.")
            except Exception as ex:
                st.error(f"Simulation error: {ex}")

    st.markdown('<div class="disclaimer">⚠️ All insights are statistical associations from a fictional dataset. Consult qualified HR professionals before any employment decision.</div>', unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ANALYTICS PAGE                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _page_analytics() -> None:
    st.markdown('<div class="page-title">🏢 Team & Department Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Aggregated attrition insights · Groups < 5 suppressed for privacy</div>', unsafe_allow_html=True)

    if not _DATA_PATH.exists():
        st.error("Dataset not found at expected path."); return

    import pandas as pd, plotly.express as px
    df = pd.read_csv(_DATA_PATH)
    df["Attrition_n"] = (df["Attrition"]=="Yes").astype(int)

    # Filters
    fc1, fc2 = st.columns(2)
    with fc1:
        dept_filter = st.multiselect("Department", df["Department"].unique().tolist(), default=df["Department"].unique().tolist())
    with fc2:
        role_filter = st.multiselect("Job Role", df["JobRole"].unique().tolist(), default=df["JobRole"].unique().tolist())
    fdf = df[df["Department"].isin(dept_filter) & df["JobRole"].isin(role_filter)]

    if len(fdf) < 5:
        st.warning("⚠️ Group size < 5. Suppressed for privacy."); return

    # KPIs
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total Employees", len(fdf))
    k2.metric("Attrition Count", int(fdf["Attrition_n"].sum()))
    k3.metric("Attrition Rate", f"{fdf['Attrition_n'].mean():.1%}")
    k4.metric("Avg Monthly Income", f"${fdf['MonthlyIncome'].mean():,.0f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Attrition Rate by Department")
        da = fdf.groupby("Department").agg(n=("Attrition_n","count"),rate=("Attrition_n","mean")).reset_index()
        da = da[da["n"]>=5]; da["rate_pct"] = (da["rate"]*100).round(1)
        fig = px.bar(da, x="Department", y="rate_pct", color="rate_pct",
                     color_continuous_scale="RdYlGn_r", template="plotly_dark",
                     labels={"rate_pct":"Attrition %"})
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Attrition by Job Role")
        ra = fdf.groupby("JobRole").agg(n=("Attrition_n","count"),rate=("Attrition_n","mean")).reset_index()
        ra = ra[ra["n"]>=5]; ra["rate_pct"]=(ra["rate"]*100).round(1)
        fig2 = px.bar(ra.sort_values("rate_pct"), x="rate_pct", y="JobRole", orientation="h",
                      color="rate_pct", color_continuous_scale="RdYlGn_r", template="plotly_dark")
        fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    # Satisfaction heatmap
    st.markdown("#### Satisfaction Scores by Department (avg)")
    sat_cols = ["JobSatisfaction","EnvironmentSatisfaction","WorkLifeBalance","RelationshipSatisfaction"]
    sat_df = fdf.groupby("Department")[sat_cols].mean().round(2)
    st.dataframe(sat_df.style.background_gradient(cmap="RdYlGn", vmin=1, vmax=4), use_container_width=True)

    # Income distribution
    st.markdown("#### Monthly Income Distribution by Job Level")
    fig3 = px.box(fdf, x="JobLevel", y="MonthlyIncome", color="Attrition",
                  color_discrete_map={"Yes":"#ef4444","No":"#10b981"},
                  template="plotly_dark")
    fig3.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="disclaimer">Data: IBM HR Analytics (fictional / synthetic). Groups < 5 suppressed for privacy.</div>', unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  BATCH SCORING PAGE                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

REQUIRED_COLS = [
    "Age","BusinessTravel","DailyRate","Department","DistanceFromHome","Education",
    "EducationField","EnvironmentSatisfaction","Gender","HourlyRate","JobInvolvement",
    "JobLevel","JobRole","JobSatisfaction","MaritalStatus","MonthlyIncome","MonthlyRate",
    "NumCompaniesWorked","OverTime","PercentSalaryHike","PerformanceRating",
    "RelationshipSatisfaction","StockOptionLevel","TotalWorkingYears",
    "TrainingTimesLastYear","WorkLifeBalance","YearsAtCompany","YearsInCurrentRole",
    "YearsSinceLastPromotion","YearsWithCurrManager",
]

def _page_batch() -> None:
    st.markdown('<div class="page-title">📁 Batch Employee Scoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Upload a CSV · Get bulk attrition predictions · Download results</div>', unsafe_allow_html=True)

    if st.session_state.get("role","viewer") not in ("admin","hr"):
        st.warning("🔒 Batch scoring requires HR Analyst or Admin role.")
        return
    if not _model_ready():
        st.error("❌ No model found. Run `python -m attrition.train` first."); return

    import pandas as pd
    template_csv = pd.DataFrame(columns=REQUIRED_COLS).to_csv(index=False).encode()
    st.download_button("⬇️ Download CSV Template", template_csv, "attrition_template.csv", "text/csv")

    uploaded = st.file_uploader("📂 Upload Employee CSV (max 500 rows)", type=["csv"])
    if not uploaded: return

    try:
        df = pd.read_csv(uploaded)
    except Exception as ex:
        st.error(f"Cannot parse CSV: {ex}"); return

    if len(df) > 500:
        st.error(f"File has {len(df)} rows. Max: 500"); return

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}"); return

    st.info(f"✅ {len(df)} rows loaded. Running predictions…")
    try:
        pipeline, meta = _load_model()
        from attrition.predict import predict_batch
        res = predict_batch(pipeline, meta, df[REQUIRED_COLS])

        nh = int((res["risk_band"]=="High").sum())
        nm = int((res["risk_band"]=="Moderate").sum())
        nl = int((res["risk_band"]=="Low").sum())
        b1,b2,b3 = st.columns(3)
        b1.metric("🔴 High Risk", nh)
        b2.metric("🟡 Moderate Risk", nm)
        b3.metric("🟢 Low Risk", nl)

        disp = ["attrition_probability","predicted_class","risk_band","model_version"]
        if "JobRole" in res.columns: disp = ["JobRole","Department"] + disp
        st.dataframe(res[disp].head(100), use_container_width=True)
        st.download_button("⬇️ Download Full Results", res.to_csv(index=False).encode(), "predictions.csv","text/csv", type="primary")
        st.markdown('<div class="disclaimer">⚠️ Results are estimates. No PII stored.</div>', unsafe_allow_html=True)
    except Exception as ex:
        st.error(f"Batch prediction failed: {ex}")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  HISTORY PAGE                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _page_history() -> None:
    st.markdown('<div class="page-title">📜 Prediction History</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Session-level prediction log</div>', unsafe_allow_html=True)

    result = st.session_state.get("last_prediction")
    emp    = st.session_state.get("employee_data")

    if result and emp:
        import pandas as pd
        row = {
            "Timestamp": result.get("timestamp","—")[:19],
            "JobRole": emp.get("JobRole","—"),
            "Department": emp.get("Department","—"),
            "Probability": f"{result.get('probability',0):.1%}",
            "Risk Band": result.get("risk_band","—"),
            "Prediction": "Leave" if result.get("class")==1 else "Stay",
            "Threshold": result.get("threshold","—"),
            "Model": result.get("model_version","v1"),
        }
        st.dataframe(pd.DataFrame([row]), use_container_width=True)
        st.info("ℹ️ Full persistent history requires the database history module (Phase 5).")
    else:
        st.info("No predictions made in this session yet.")
        if st.button("Go to Predict →", type="primary"):
            st.session_state.page = "predict"; st.rerun()

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ADMIN PAGE                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _page_admin() -> None:
    if st.session_state.get("role") != "admin":
        st.error("🔒 Admin access required."); return
    st.markdown('<div class="page-title">⚙️ Admin Panel</div>', unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["👥 Users", "🤖 Model Info", "🔧 Config"])

    with t1:
        try:
            import pandas as pd
            conn = sqlite3.connect(str(_DB_PATH))
            users = pd.read_sql("SELECT username, role, created_at FROM users", conn); conn.close()
            st.dataframe(users, use_container_width=True)
        except Exception as ex:
            st.error(str(ex))

        with st.form("add_user"):
            st.markdown("#### ➕ Add New User")
            nu = st.text_input("Username")
            np_ = st.text_input("Password", type="password")
            nr = st.selectbox("Role", ["viewer","hr","admin"])
            if st.form_submit_button("Add User"):
                conn = sqlite3.connect(str(_DB_PATH))
                try:
                    conn.execute("INSERT INTO users (username,password_hash,role,created_at) VALUES (?,?,?,?)",
                                 (nu.lower(), _hash_pw(np_), nr, datetime.now(timezone.utc).isoformat()))
                    conn.commit(); st.success(f"User '{nu}' created.")
                except Exception as ex:
                    st.error(str(ex))
                finally:
                    conn.close()

    with t2:
        meta = _load_meta()
        if meta:
            st.json(meta)
        else:
            st.warning("No model trained yet.")

    with t3:
        from attrition.config import get_config
        cfg = get_config()
        st.json({"app_env":cfg.app_env,"model_version":cfg.model_version,
                 "random_seed":cfg.random_seed,"retention_days":cfg.prediction_log_retention_days,
                 "min_group_size":cfg.min_group_size})

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODEL HEALTH PAGE                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _page_health() -> None:
    if st.session_state.get("role") != "admin":
        st.error("🔒 Admin access required."); return
    st.markdown('<div class="page-title">❤️ Model Health Monitor</div>', unsafe_allow_html=True)

    meta = _load_meta()
    if not meta:
        st.warning("No model trained yet. Run `python -m attrition.train`."); return

    m  = meta.get("test_metrics",{})
    cv = meta.get("cv_metrics",{})

    st.markdown("### 📊 Test-Set Metrics")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("ROC-AUC",  f"{m.get('roc_auc',0):.4f}")
    c2.metric("Recall",   f"{m.get('recall',0):.4f}")
    c3.metric("Precision",f"{m.get('precision',0):.4f}")
    c4.metric("F1",       f"{m.get('f1',0):.4f}")
    c5.metric("Brier",    f"{m.get('brier_score',0):.4f}")

    if cv:
        import pandas as pd
        st.markdown("### 📈 Cross-Validation (5-fold × 3 repeats)")
        cv_df = pd.DataFrame({k:{"Mean":v["mean"],"Std":v["std"]} for k,v in cv.items()}).T.round(4)
        st.dataframe(cv_df, use_container_width=True)

    st.markdown("### 🎯 Risk Bands")
    bands = meta.get("risk_bands",{})
    for bname, binfo in bands.items():
        if isinstance(binfo, dict) and "observed_attrition_rate" in binfo:
            obs = binfo["observed_attrition_rate"]
            st.markdown(f"**{bname.capitalize()}**: Observed attrition rate = **{obs:.1%}**")

    st.markdown("### 📑 Full Metadata")
    st.json({k:v for k,v in meta.items() if k not in ("test_metrics","cv_metrics","risk_bands")})

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ROUTER                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main() -> None:
    _ensure_db()
    _init_session()

    if not st.session_state.authenticated:
        _page_login()
        return

    _sidebar()

    page = st.session_state.get("page","home")
    pages = {
        "home":      _page_home,
        "predict":   _page_predict,
        "insights":  _page_insights,
        "analytics": _page_analytics,
        "batch":     _page_batch,
        "history":   _page_history,
        "admin":     _page_admin,
        "health":    _page_health,
    }
    fn = pages.get(page, _page_home)
    fn()


if __name__ == "__main__":
    main()
