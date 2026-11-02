# Architecture Decision Records (ADR) & Technical Decisions

**Project**: Real-Time Employee Attrition Prediction Dashboard with Intelligent Insights  
**Lead**: Staff ML Engineer & MLOps Lead

---

## ADR-001: Separation of Concerns (Library vs UI)
- **Status**: Accepted
- **Context**: Streamlit applications often suffer from tight coupling of UI logic, database access, and ML training code.
- **Decision**: All machine learning, feature extraction, evaluation, calibration, fairness, security, and persistence logic resides in `src/attrition/`. The `app/` directory contains purely presentation and view components using Streamlit.
- **Consequences**: Enables clean unit/integration testing without running Streamlit UI, headless batch prediction, and future migration to FastAPI or other frontend frameworks.

---

## ADR-002: Categorical Feature Encoding Strategy
- **Status**: Accepted
- **Context**: The baseline model used `OrdinalEncoder` on unordered nominal categoricals (`Department`, `JobRole`, `EducationField`, `MaritalStatus`, `BusinessTravel`, `OverTime`).
- **Decision**: Use `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` within a `ColumnTransformer` for all nominal categoricals. Genuine ordinal scale ratings (e.g., `JobSatisfaction`, `WorkLifeBalance`, `JobLevel` rated 1–4 or 1–5) are treated as numeric continuous-like intervals and standardized via `StandardScaler`.
- **Consequences**: Mathematically sound linear representation; eliminates artificial ordinal ordering.

---

## ADR-003: Fairness and Protected Attributes (Removal of Gender)
- **Status**: Accepted
- **Context**: `Gender` appeared in the top-20 baseline features. Employment decisions cannot lawfully or ethically be influenced by gender.
- **Decision**: Conduct an explicit fairness audit using `fairlearn`. Train a model variant excluding `Gender` as an input feature. If performance drop is negligible (ROC-AUC / F1 delta < 0.01), deploy the Gender-free model to production. Retain the gender field in the UI form only as an optional demographic field for aggregate fairness monitoring, never fed into the inference pipeline.
- **Consequences**: Zero algorithmic gender bias in attrition risk scores. Compliance with ethical AI and EEOC / EU AI Act principles.

---

## ADR-004: Probability Calibration and Empirical Risk Bands
- **Status**: Accepted
- **Context**: Raw classifier probabilities often deviate from true empirical frequencies, leading to misleading risk categorizations.
- **Decision**: Use `CalibratedClassifierCV` (evaluating both Sigmoid and Isotonic calibration via Brier score). Derive Low / Moderate / High risk bands based on calibrated probability thresholds tied to empirical attrition prevalence:
  - **Low Risk**: $P(\text{Attrition}) < 0.20$ (observed attrition rate $< 5\%$)
  - **Moderate Risk**: $0.20 \le P(\text{Attrition}) < 0.45$
  - **High Risk**: $P(\text{Attrition}) \ge 0.45$
- **Consequences**: Transparent and reliable risk banding; provides decision-makers with truthful empirical risk indicators.

---

## ADR-005: Local Explainability via SHAP
- **Status**: Accepted
- **Context**: Global feature importance fails to explain why a specific individual was flagged as high risk.
- **Decision**: Implement per-employee SHAP (`LinearExplainer` for Logistic Regression; `TreeExplainer` for trees). Display directional waterfall / bar insights (factors increasing vs. decreasing risk) with explicit disclaimers separating association from causation.
- **Consequences**: Interpretable, actionable insights for HR business partners without causal overreach.

---

## ADR-006: Security & Privacy by Design
- **Status**: Accepted
- **Context**: HR analytics deals with sensitive employee data.
- **Decision**:
  1. Argon2id password hashing for credential storage.
  2. Rate limiting and account lockout after 5 consecutive failed attempts (15-minute window).
  3. Prediction logs store only cryptographic input fingerprints (`SHA256(data + salt)`), timestamps, predicted risk bands, and model versions — never plaintext PII.
  4. Team analytics enforce $k$-anonymity threshold ($k \ge 5$ employees) before displaying group metrics.
- **Consequences**: Complies with GDPR / data privacy requirements and enterprise security benchmarks.
