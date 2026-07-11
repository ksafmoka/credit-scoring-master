"""Pydantic schemas for the scoring API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ScoringRequest(BaseModel):
    application_id: int
    loan_amount: float = Field(gt=0)
    income: float = Field(gt=0)
    loan_term: int = Field(ge=12, le=84)
    interest_rate: float = Field(ge=0, le=50)
    employment_years: float = Field(ge=0, le=50)
    credit_score: int = Field(ge=300, le=850)
    dti_ratio: float = Field(ge=0, le=100)
    num_open_accounts: int = Field(ge=0)
    num_delinquencies: int = Field(ge=0)
    total_credit_limit: float = Field(ge=0)
    home_ownership: str
    purpose: str

    # Optional online signals (default to 0 / missing when unknown)
    avg_days_overdue_30d: float | None = None
    avg_days_overdue_90d: float | None = None
    avg_days_overdue_180d: float | None = None
    max_days_overdue_90d: float | None = None
    pct_late_payments_90d: float | None = None
    total_paid_90d: float | None = None
    payment_consistency_90d: float | None = None
    bureau_balance_to_income: float | None = None
    inquiries_per_account: float | None = None

    @field_validator("home_ownership")
    @classmethod
    def validate_home_ownership(cls, v: str) -> str:
        allowed = {"RENT", "OWN", "MORTGAGE", "OTHER"}
        if v.upper() not in allowed:
            raise ValueError(f"home_ownership must be one of {allowed}")
        return v.upper()

    @field_validator("purpose")
    @classmethod
    def normalize_purpose(cls, v: str) -> str:
        return v.strip().lower()


class ReasonCode(BaseModel):
    feature: str
    shap_value: float
    direction: str


class ScoringResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    application_id: int
    pd_score: float
    pd_calibrated: float
    risk_bucket: str
    top_reasons: list[ReasonCode]
    model_version: str
