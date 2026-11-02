# AS-IS Architecture Audit & Baseline State Analysis

**Project**: Real-Time Employee Attrition Prediction Dashboard with Intelligent Insights  
**Laboratory**: Advanced Data Science Laboratory (PEIT-412B), Sanjivani College of Engineering  
**Authors**: Kadam Chetan Shivaji, Kale Nikhil Dattatray, Kandalkar Suraj Balu, Khemnar Om Dipak, Dabhade Yash Vijay  
**Date**: November 2024 / Updated November 2026  
**Auditor**: Staff ML & MLOps Lead

---

## 1. Executive Summary

This document captures the baseline architectural and methodological state of the Employee Attrition Prediction Dashboard as evaluated from the original academic project report and prototype specification, prior to production industrialization.

The baseline prototype demonstrated proof-of-concept feasibility for real-time employee attrition prediction using the IBM HR Analytics dataset, deployed via Streamlit on Hugging Face Spaces (Docker port 7860). However, critical methodological, statistical, fairness, security, and architectural gaps prevent enterprise deployment.

---

## 2. Dataset & Features

- **Source**: IBM HR Analytics Employee Attrition & Performance dataset (synthetic/fictional benchmark).
- **Scale**: 1,470 employee records across 35 attributes.
- **Target Variable**: `Attrition` ("Yes" vs "No"), positive prevalence 16.12% (237 leavers, 1,233 stayers) — pronounced class imbalance (~5.2:1).
- **Constant / Zero-variance Features Dropped**:
  - `EmployeeCount` (constant value: 1)
  - `EmployeeNumber` (arbitrary identifier)
  - `Over18` (constant value: 'Y')
  - `StandardHours` (constant value: 80)
- **Active Feature Space**: 30 features (23 numeric/ordinal scales, 7 nominal categoricals).

---

## 3. Baseline Modeling & Evaluation Architecture

### 3.1 Preprocessing Pipeline Flaws
1. **Arbitrary Ordinal Encoding on Nominal Categoricals**: Nominal features with no natural order (e.g. `Department`, `EducationField`, `JobRole`) were encoded using `OrdinalEncoder`. For linear models (Logistic Regression), this imposes an artificial, spurious Euclidean ordering and distance metric.
2. **Data Leakage Risk**: Feature scaling (`StandardScaler`) was not strictly encapsulated in an immutable scikit-learn Pipeline prior to train/test splitting.
3. **Imbalance Handling**: The baseline did not use class weighting, SMOTE, or cost-sensitive threshold adjustments.

### 3.2 Reported Baseline Metrics (Logistic Regression, 294-row 80/20 Test Split)
- **Accuracy**: 0.7517 (75.17%)
- **Precision**: 0.3617 (36.17%)
- **Recall**: 0.7234 (72.34%)
- **F1-Score**: 0.4823 (48.23%)
- **ROC-AUC**: 0.8051

### 3.3 Known Flaws & Limitations in Baseline
1. **Uncalibrated Probabilities**: Raw logistic regression sigmoid outputs were treated as true posterior probabilities without calibration (`CalibratedClassifierCV`). Risk bands (Low / Moderate / High) were arbitrary percentage cutoffs rather than empirical, precision-calibrated thresholds.
2. **Protected Attribute Leakage**: `Gender` was retained as an input feature and ranked among the top-20 influential coefficients without fairness auditing (disparate impact, demographic parity, equal opportunity).
3. **Single Split Evaluation**: Metrics were reported from a single 80/20 test split (only 47 positive attrition instances in the test set) without cross-validation or bootstrap confidence intervals.
4. **No Local Explainability**: Only global logistic regression weights were reported; no per-employee SHAP explanations were generated.
5. **Missing Product & MLOps Features**:
   - No prediction history logging or privacy-preserving audit logs.
   - No exportable executive reports (PDF/HTML).
   - No team/department aggregation or minimum privacy group sizing ($k \ge 5$).
   - Table 16 test cases 3–10 remained "Pending".
   - Weak authentication (no rate limiting or Argon2 hashing).
   - Technical diagnostics were exposed to standard end users.

---

## 4. Planned Industrialization Target

The refactoring will transition the repository to a modular production structure (`src/attrition/`), robust Pydantic schemas, unified scikit-learn ColumnTransformer pipelines, repeated stratified 5-fold cross-validation with Optuna hyperparameter optimization, fairlearn bias auditing, SHAP explainability, Argon2 security, and multi-stage Docker deployment.
