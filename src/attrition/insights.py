"""Employee insight generation — profile summaries, recommendations, and risk analysis.

Provides rule-based recommendation logic and natural-language summaries
derived from employee attributes and SHAP-based explanations.
"""

from __future__ import annotations

from typing import Any

from .logging_conf import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Recommendation rules table
# ---------------------------------------------------------------------------

RECOMMENDATION_RULES: list[dict[str, Any]] = [
    # OverTime
    {
        "condition": lambda e: str(e.get("OverTime", "No")).lower() in ("yes", "1", "true"),
        "category": "Work-Life Balance",
        "recommendation": "Review and reduce overtime hours. Consider workload redistribution or hiring additional support.",
        "priority": "high",
        "rationale": "Employees working overtime are significantly more likely to leave the organisation.",
    },
    # JobSatisfaction
    {
        "condition": lambda e: e.get("JobSatisfaction", 3) <= 2,
        "category": "Job Satisfaction",
        "recommendation": "Conduct a stay interview to understand job dissatisfaction drivers. Explore role enrichment, lateral moves, or stretch projects.",
        "priority": "high",
        "rationale": "Low job satisfaction (≤2) is a strong predictor of voluntary attrition.",
    },
    # WorkLifeBalance
    {
        "condition": lambda e: e.get("WorkLifeBalance", 3) <= 2,
        "category": "Work-Life Balance",
        "recommendation": "Offer flexible work arrangements, remote work options, or adjusted schedules.",
        "priority": "high",
        "rationale": "Poor work-life balance is closely associated with employee turnover.",
    },
    # YearsSinceLastPromotion
    {
        "condition": lambda e: e.get("YearsSinceLastPromotion", 0) >= 5,
        "category": "Career Growth",
        "recommendation": "Discuss career progression with the employee. Consider a promotion review or create a clear development plan with milestones.",
        "priority": "high",
        "rationale": "Employees who have not been promoted in 5+ years are at elevated attrition risk.",
    },
    # MonthlyIncome (relative to common thresholds)
    {
        "condition": lambda e: e.get("MonthlyIncome", 10000) < 3000,
        "category": "Compensation",
        "recommendation": "Benchmark the employee's compensation against market rates and consider a pay adjustment or performance bonus.",
        "priority": "high",
        "rationale": "Below-market compensation drives attrition, especially for lower-income employees.",
    },
    # DistanceFromHome
    {
        "condition": lambda e: e.get("DistanceFromHome", 0) >= 20,
        "category": "Work-Life Balance",
        "recommendation": "Offer remote or hybrid work options, commuting benefits, or explore relocation assistance.",
        "priority": "medium",
        "rationale": "Long commutes (≥20 units) contribute to dissatisfaction and increase attrition risk.",
    },
    # TrainingTimesLastYear
    {
        "condition": lambda e: e.get("TrainingTimesLastYear", 3) <= 1,
        "category": "Learning & Development",
        "recommendation": "Increase training opportunities — enrol the employee in relevant courses, certifications, or mentoring programmes.",
        "priority": "medium",
        "rationale": "Limited training may signal lack of career investment and increase flight risk.",
    },
    # EnvironmentSatisfaction
    {
        "condition": lambda e: e.get("EnvironmentSatisfaction", 3) <= 2,
        "category": "Work Environment",
        "recommendation": "Investigate environmental pain-points (workspace, tools, team dynamics) and take corrective action.",
        "priority": "medium",
        "rationale": "Low environment satisfaction is linked to higher turnover intention.",
    },
    # RelationshipSatisfaction
    {
        "condition": lambda e: e.get("RelationshipSatisfaction", 3) <= 2,
        "category": "Team Relationships",
        "recommendation": "Facilitate team-building activities and ensure the employee has access to conflict resolution support.",
        "priority": "medium",
        "rationale": "Poor workplace relationships reduce engagement and increase attrition risk.",
    },
    # NumCompaniesWorked (high job-hopping)
    {
        "condition": lambda e: e.get("NumCompaniesWorked", 1) >= 7,
        "category": "Retention Strategy",
        "recommendation": "Proactively engage the employee with retention incentives — stock options, retention bonuses, or personalised career paths.",
        "priority": "medium",
        "rationale": "Employees with many prior employers may have a higher baseline propensity to leave.",
    },
    # YearsAtCompany (very early tenure)
    {
        "condition": lambda e: e.get("YearsAtCompany", 5) <= 1,
        "category": "Onboarding",
        "recommendation": "Ensure a robust onboarding experience with a buddy programme, regular check-ins, and clear 90-day goals.",
        "priority": "medium",
        "rationale": "New hires (≤1 year) are at elevated attrition risk during the adjustment period.",
    },
    # Age (younger employees)
    {
        "condition": lambda e: e.get("Age", 35) < 28,
        "category": "Career Growth",
        "recommendation": "Offer mentoring, skill-development programmes, and visible growth opportunities to retain younger talent.",
        "priority": "low",
        "rationale": "Younger employees tend to have higher mobility and may leave for career advancement.",
    },
    # StockOptionLevel
    {
        "condition": lambda e: e.get("StockOptionLevel", 1) == 0,
        "category": "Compensation",
        "recommendation": "Consider introducing equity-based incentives or profit-sharing to strengthen long-term retention.",
        "priority": "low",
        "rationale": "Employees with no stock options have less financial incentive to stay.",
    },
    # JobInvolvement
    {
        "condition": lambda e: e.get("JobInvolvement", 3) <= 2,
        "category": "Engagement",
        "recommendation": "Assign more meaningful and challenging work. Involve the employee in decision-making and cross-functional projects.",
        "priority": "medium",
        "rationale": "Low job involvement indicates disengagement, a key precursor to attrition.",
    },
]


def generate_profile_summary(employee: dict[str, Any]) -> str:
    """Create a human-readable textual summary of an employee profile.

    Args:
        employee: Dictionary of employee attributes (feature names as keys).

    Returns:
        Multi-line string summarising key employee details.
    """
    log.debug("generate_profile_summary.start")

    age = employee.get("Age", "N/A")
    gender = employee.get("Gender", "N/A")
    department = employee.get("Department", "N/A")
    job_role = employee.get("JobRole", "N/A")
    job_level = employee.get("JobLevel", "N/A")
    years_at_company = employee.get("YearsAtCompany", "N/A")
    monthly_income = employee.get("MonthlyIncome", "N/A")
    overtime = employee.get("OverTime", "N/A")
    marital_status = employee.get("MaritalStatus", "N/A")
    education = employee.get("Education", "N/A")
    education_field = employee.get("EducationField", "N/A")
    total_working_years = employee.get("TotalWorkingYears", "N/A")
    years_since_promotion = employee.get("YearsSinceLastPromotion", "N/A")
    distance = employee.get("DistanceFromHome", "N/A")
    num_companies = employee.get("NumCompaniesWorked", "N/A")

    education_map = {1: "Below College", 2: "College", 3: "Bachelor", 4: "Master", 5: "Doctor"}
    edu_label = education_map.get(education, str(education))

    lines = [
        f"Employee Profile Summary",
        f"========================",
        f"Demographics: {age}-year-old {gender}, {marital_status}",
        f"Education:    {edu_label} in {education_field}",
        f"Role:         {job_role} (Level {job_level}) in {department}",
        f"Tenure:       {years_at_company} years at company, {total_working_years} years total experience",
        f"Compensation: ${monthly_income:,}/month" if isinstance(monthly_income, (int, float)) else f"Compensation: {monthly_income}",
        f"Overtime:     {overtime}",
        f"Commute:      {distance} units from home",
        f"Mobility:     {num_companies} previous companies",
        f"Last promotion: {years_since_promotion} years ago",
    ]

    summary = "\n".join(lines)
    log.debug("generate_profile_summary.done", summary_length=len(summary))
    return summary


def generate_recommendations(
    employee: dict[str, Any],
    risk_band: str,
    top_factors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Generate actionable recommendations based on employee data and risk.

    Evaluates each entry in :data:`RECOMMENDATION_RULES` against the
    employee's attributes and returns applicable recommendations,
    sorted by priority.

    Args:
        employee: Employee attribute dictionary.
        risk_band: Current risk band (``"Low"``, ``"Moderate"``, or
            ``"High"``).
        top_factors: Top contributing factors from
            :func:`~attrition.explain.get_top_factors`.

    Returns:
        List of recommendation dicts::

            [
                {
                    "category": str,
                    "recommendation": str,
                    "priority": "high" | "medium" | "low",
                    "rationale": str,
                },
                ...
            ]
    """
    log.info(
        "generate_recommendations.start",
        risk_band=risk_band,
        n_factors=len(top_factors),
    )

    priority_order = {"high": 0, "medium": 1, "low": 2}
    recommendations: list[dict[str, Any]] = []

    # Gather risk-increasing feature names for boosting priority
    risk_feature_names = {
        f["feature"]
        for f in top_factors
        if f.get("direction") == "increases risk"
    }

    for rule in RECOMMENDATION_RULES:
        try:
            if rule["condition"](employee):
                rec = {
                    "category": rule["category"],
                    "recommendation": rule["recommendation"],
                    "priority": rule["priority"],
                    "rationale": rule["rationale"],
                }

                # Boost priority if the rule's category aligns with a top risk factor
                if risk_band == "High" and rec["priority"] == "medium":
                    rec["priority"] = "high"

                recommendations.append(rec)
        except Exception as exc:
            log.warning(
                "generate_recommendations.rule_error",
                category=rule.get("category"),
                error=str(exc),
            )

    # Sort by priority
    recommendations.sort(key=lambda r: priority_order.get(r["priority"], 99))

    log.info(
        "generate_recommendations.done",
        n_recommendations=len(recommendations),
    )
    return recommendations


def get_strengths(
    employee: dict[str, Any],
    top_factors: list[dict[str, Any]],
) -> list[str]:
    """Identify positive (risk-decreasing) factors for an employee.

    Args:
        employee: Employee attribute dictionary.
        top_factors: Top contributing factors from SHAP analysis.

    Returns:
        List of human-readable strength descriptions.
    """
    log.debug("get_strengths.start")

    strengths: list[str] = []
    feature_label_map = _build_feature_label_map(employee)

    for factor in top_factors:
        if factor.get("direction") == "decreases risk":
            feature = factor["feature"]
            label = feature_label_map.get(feature, feature)
            contribution = abs(factor.get("contribution", 0))
            strengths.append(
                f"{label} (protective effect: {contribution:.4f})"
            )

    # Add generic strengths based on employee attributes
    if employee.get("YearsAtCompany", 0) >= 10:
        strengths.append("Long company tenure (10+ years) indicates strong organisational commitment")
    if employee.get("JobSatisfaction", 0) >= 4:
        strengths.append("High job satisfaction (4/4)")
    if employee.get("WorkLifeBalance", 0) >= 4:
        strengths.append("Excellent work-life balance rating (4/4)")
    if employee.get("EnvironmentSatisfaction", 0) >= 4:
        strengths.append("High environment satisfaction (4/4)")
    if employee.get("RelationshipSatisfaction", 0) >= 4:
        strengths.append("Strong workplace relationships (4/4)")
    if employee.get("StockOptionLevel", 0) >= 2:
        strengths.append("Meaningful equity stake (stock option level ≥2)")

    # De-duplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for s in strengths:
        if s not in seen:
            seen.add(s)
            unique.append(s)

    log.debug("get_strengths.done", n_strengths=len(unique))
    return unique


def get_risk_factors(
    employee: dict[str, Any],
    top_factors: list[dict[str, Any]],
) -> list[str]:
    """Identify negative (risk-increasing) factors for an employee.

    Args:
        employee: Employee attribute dictionary.
        top_factors: Top contributing factors from SHAP analysis.

    Returns:
        List of human-readable risk-factor descriptions.
    """
    log.debug("get_risk_factors.start")

    risks: list[str] = []
    feature_label_map = _build_feature_label_map(employee)

    for factor in top_factors:
        if factor.get("direction") == "increases risk":
            feature = factor["feature"]
            label = feature_label_map.get(feature, feature)
            contribution = abs(factor.get("contribution", 0))
            risks.append(
                f"{label} (risk contribution: {contribution:.4f})"
            )

    # Add generic risk flags based on employee attributes
    if str(employee.get("OverTime", "No")).lower() in ("yes", "1", "true"):
        risks.append("Currently working overtime")
    if employee.get("JobSatisfaction", 3) <= 2:
        risks.append(f"Low job satisfaction ({employee.get('JobSatisfaction')}/4)")
    if employee.get("WorkLifeBalance", 3) <= 2:
        risks.append(f"Poor work-life balance ({employee.get('WorkLifeBalance')}/4)")
    if employee.get("YearsSinceLastPromotion", 0) >= 5:
        risks.append(f"No promotion in {employee.get('YearsSinceLastPromotion')} years")
    if employee.get("MonthlyIncome", 10000) < 3000:
        risks.append(f"Below-average compensation (${employee.get('MonthlyIncome'):,}/month)")

    # De-duplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for r in risks:
        if r not in seen:
            seen.add(r)
            unique.append(r)

    log.debug("get_risk_factors.done", n_risks=len(unique))
    return unique


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_feature_label_map(employee: dict[str, Any]) -> dict[str, str]:
    """Create a mapping from feature names to readable labels with values."""
    label_map: dict[str, str] = {}
    readable = {
        "OverTime": "Overtime",
        "JobSatisfaction": "Job Satisfaction",
        "WorkLifeBalance": "Work-Life Balance",
        "EnvironmentSatisfaction": "Environment Satisfaction",
        "RelationshipSatisfaction": "Relationship Satisfaction",
        "JobInvolvement": "Job Involvement",
        "MonthlyIncome": "Monthly Income",
        "YearsSinceLastPromotion": "Years Since Last Promotion",
        "YearsAtCompany": "Years at Company",
        "YearsInCurrentRole": "Years in Current Role",
        "YearsWithCurrManager": "Years with Current Manager",
        "TotalWorkingYears": "Total Working Years",
        "DistanceFromHome": "Distance from Home",
        "NumCompaniesWorked": "Num Companies Worked",
        "Age": "Age",
        "DailyRate": "Daily Rate",
        "HourlyRate": "Hourly Rate",
        "MonthlyRate": "Monthly Rate",
        "PercentSalaryHike": "Percent Salary Hike",
        "TrainingTimesLastYear": "Training Times Last Year",
        "Education": "Education",
        "StockOptionLevel": "Stock Option Level",
        "JobLevel": "Job Level",
        "PerformanceRating": "Performance Rating",
    }

    for feature, label in readable.items():
        value = employee.get(feature)
        if value is not None:
            label_map[feature] = f"{label} = {value}"
        else:
            label_map[feature] = label

    return label_map
