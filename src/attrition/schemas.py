"""Pydantic input validation schemas for employee data.

The :class:`EmployeeInput` model validates all 30 features (after the
four constant columns are dropped) with proper types, value ranges, and
cross-field consistency checks.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, model_validator


class EmployeeInput(BaseModel):
    """Validated input representing a single employee record.

    Every field corresponds to a column in the IBM HR Analytics dataset
    (after dropping ``EmployeeCount``, ``EmployeeNumber``, ``Over18``,
    and ``StandardHours``).
    """

    # ── Truly numeric features ──────────────────────────────────────────

    Age: int = Field(
        ..., ge=18, le=65, description="Employee age (18–65)"
    )
    DailyRate: int = Field(
        ..., ge=1, le=1_500, description="Daily rate in USD"
    )
    DistanceFromHome: int = Field(
        ..., ge=1, le=30, description="Distance from home in miles (1–29)"
    )
    HourlyRate: int = Field(
        ..., ge=30, le=100, description="Hourly rate in USD"
    )
    MonthlyIncome: int = Field(
        ..., ge=1_000, le=20_000, description="Monthly income in USD"
    )
    MonthlyRate: int = Field(
        ..., ge=2_000, le=27_000, description="Monthly rate"
    )
    NumCompaniesWorked: int = Field(
        ..., ge=0, le=10, description="Number of companies worked at (0–9)"
    )
    PercentSalaryHike: int = Field(
        ..., ge=10, le=25, description="Percent salary hike (11–25)"
    )
    TotalWorkingYears: int = Field(
        ..., ge=0, le=45, description="Total working years (0–40)"
    )
    TrainingTimesLastYear: int = Field(
        ..., ge=0, le=6, description="Training times last year (0–6)"
    )
    YearsAtCompany: int = Field(
        ..., ge=0, le=40, description="Years at current company (0–40)"
    )
    YearsInCurrentRole: int = Field(
        ..., ge=0, le=20, description="Years in current role (0–18)"
    )
    YearsSinceLastPromotion: int = Field(
        ..., ge=0, le=15, description="Years since last promotion (0–15)"
    )
    YearsWithCurrManager: int = Field(
        ..., ge=0, le=20, description="Years with current manager (0–17)"
    )

    # ── Ordinal-like integer scales (treated as numeric) ────────────────

    Education: int = Field(
        ...,
        ge=1,
        le=5,
        description="Education level: 1=Below College … 5=Doctor",
    )
    EnvironmentSatisfaction: int = Field(
        ...,
        ge=1,
        le=4,
        description="Environment satisfaction: 1=Low … 4=Very High",
    )
    JobInvolvement: int = Field(
        ...,
        ge=1,
        le=4,
        description="Job involvement: 1=Low … 4=Very High",
    )
    JobLevel: int = Field(
        ...,
        ge=1,
        le=5,
        description="Job level (1–5)",
    )
    JobSatisfaction: int = Field(
        ...,
        ge=1,
        le=4,
        description="Job satisfaction: 1=Low … 4=Very High",
    )
    PerformanceRating: int = Field(
        ...,
        ge=1,
        le=5,
        description="Performance rating: 1=Low … 5=Outstanding",
    )
    RelationshipSatisfaction: int = Field(
        ...,
        ge=1,
        le=4,
        description="Relationship satisfaction: 1=Low … 4=Very High",
    )
    StockOptionLevel: int = Field(
        ...,
        ge=0,
        le=3,
        description="Stock option level (0–3)",
    )
    WorkLifeBalance: int = Field(
        ...,
        ge=1,
        le=4,
        description="Work-life balance: 1=Bad … 4=Best",
    )

    # ── Nominal categorical features ────────────────────────────────────

    BusinessTravel: Literal[
        "Non-Travel", "Travel_Rarely", "Travel_Frequently"
    ] = Field(..., description="Business travel frequency")

    Department: Literal[
        "Human Resources", "Research & Development", "Sales"
    ] = Field(..., description="Department name")

    EducationField: Literal[
        "Human Resources",
        "Life Sciences",
        "Marketing",
        "Medical",
        "Other",
        "Technical Degree",
    ] = Field(..., description="Field of education")

    Gender: Literal["Female", "Male"] = Field(
        ..., description="Employee gender"
    )

    JobRole: Literal[
        "Healthcare Representative",
        "Human Resources",
        "Laboratory Technician",
        "Manager",
        "Manufacturing Director",
        "Research Director",
        "Research Scientist",
        "Sales Executive",
        "Sales Representative",
    ] = Field(..., description="Job role title")

    MaritalStatus: Literal["Divorced", "Married", "Single"] = Field(
        ..., description="Marital status"
    )

    OverTime: Literal["Yes", "No"] = Field(
        ..., description="Whether the employee works overtime"
    )

    # ── Cross-field validators ──────────────────────────────────────────

    @model_validator(mode="after")
    def _validate_year_relationships(self) -> "EmployeeInput":
        """Ensure logical consistency among year-based fields."""
        errors: list[str] = []

        if self.YearsAtCompany > self.TotalWorkingYears:
            errors.append(
                f"YearsAtCompany ({self.YearsAtCompany}) cannot exceed "
                f"TotalWorkingYears ({self.TotalWorkingYears})"
            )

        if self.TotalWorkingYears >= self.Age:
            errors.append(
                f"TotalWorkingYears ({self.TotalWorkingYears}) must be less "
                f"than Age ({self.Age})"
            )

        if self.YearsInCurrentRole > self.YearsAtCompany:
            errors.append(
                f"YearsInCurrentRole ({self.YearsInCurrentRole}) cannot "
                f"exceed YearsAtCompany ({self.YearsAtCompany})"
            )

        if self.YearsWithCurrManager > self.YearsAtCompany:
            errors.append(
                f"YearsWithCurrManager ({self.YearsWithCurrManager}) cannot "
                f"exceed YearsAtCompany ({self.YearsAtCompany})"
            )

        if self.YearsSinceLastPromotion > self.YearsAtCompany:
            errors.append(
                f"YearsSinceLastPromotion ({self.YearsSinceLastPromotion}) "
                f"cannot exceed YearsAtCompany ({self.YearsAtCompany})"
            )

        if errors:
            raise ValueError(
                "Cross-field validation failed:\n  • " + "\n  • ".join(errors)
            )

        return self

    # ── Conversion ──────────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        """Convert this validated input to a single-row DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with one row and 30 columns matching the model
            feature order.
        """
        return pd.DataFrame([self.model_dump()])
