# src/serving/schemas.py

from pydantic import BaseModel, Field, field_validator
from typing import Optional


class ScoringRequest(BaseModel):
    application_id: int
    loan_amount: float = Field(gt=0, description="Loan amount in USD")
    income: float = Field(gt=0, description="Annual income in USD")
    loan_term: int = Field(
        ge=12, le=84, description="Loan term in months"
    )
    interest_rate: float = Field(ge=0, le=50)
    employment_years: float = Field(ge=0, le=50)
    credit_score: int = Field(ge=300, le=850)
    dti_ratio: float = Field(ge=0, le=100)
    num_open_accounts: int = Field(ge=0)
    num_delinquencies: int = Field(ge=0)
    total_credit_limit: float = Field(ge=0)
    home_ownership: str
    purpose: str

    @field_validator("home_ownership")
    @classmethod
    def validate_home_ownership(cls, v: str) -> str:
        allowed = {"RENT", "OWN", "MORTGAGE", "OTHER"}
        if v.upper() not in allowed:
            raise ValueError(
                f"home_ownership must be one of {allowed}"
            )
        return v.upper()


class ReasonCode(BaseModel):
    feature: str
    shap_value: float
    direction: str


class ScoringResponse(BaseModel):
    application_id: int
    pd_score: float = Field(
        description="Raw probability of default"
    )
    pd_calibrated: float = Field(
        description="Calibrated probability of default"
    )
    risk_bucket: str = Field(
        description="LOW / MEDIUM / HIGH / VERY_HIGH"
    )
    top_reasons: list[ReasonCode]
    model_version: str