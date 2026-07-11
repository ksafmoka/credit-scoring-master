"""Central configuration for the credit scoring system."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT_DIR / "configs"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


class DBConfig:
    HOST: str = os.getenv("APP_DB_HOST", "localhost")
    PORT: int = int(os.getenv("APP_DB_PORT", "5432"))
    NAME: str = os.getenv("APP_DB_NAME", "credit_scoring")
    USER: str = os.getenv("APP_DB_USER", "ml_user")
    PASSWORD: str = os.getenv("APP_DB_PASSWORD", "ml_password")

    @classmethod
    def get_url(cls) -> str:
        return (
            f"postgresql+psycopg2://{cls.USER}:{cls.PASSWORD}"
            f"@{cls.HOST}:{cls.PORT}/{cls.NAME}"
        )


class MLflowConfig:
    TRACKING_URI: str = os.getenv(
        "MLFLOW_TRACKING_URI", "http://localhost:5000"
    )
    EXPERIMENT_SCORING: str = "credit_scoring"
    REGISTERED_MODEL_NAME: str = "credit_scoring"
    MODEL_ALIAS: str = "champion"


class TrainingConfig:
    TRAIN_END_DATE: str = "2022-12-31"
    VAL_END_DATE: str = "2023-06-30"
    RANDOM_SEED: int = 42
    N_OPTUNA_TRIALS: int = 15
    CV_FOLDS: int = 5
    TARGET_COL: str = "is_default"
    ID_COL: str = "application_id"
    DATE_COL: str = "application_date"
    MIN_AUC_FOR_REGISTRATION: float = 0.70
    CALIBRATION_METHOD: str = "isotonic"  # isotonic | sigmoid | none


class MonitoringConfig:
    PSI_THRESHOLD: float = 0.15
    PSI_WARNING: float = 0.10
    AUC_DROP_THRESHOLD: float = 0.03
    MIN_PREDICTIONS_FOR_CHECK: int = 100
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")


class FeatureConfig:
    VERSION: str = "v1.0"
    PAYMENT_WINDOWS: list[int] = [30, 90, 180]

    # Application-level engineered numerics + cross features
    NUMERICAL_FEATURES: list[str] = [
        "loan_to_income",
        "credit_utilization",
        "income_log",
        "loan_amount_log",
        "dti_ratio_clipped",
        "employment_years",
        "credit_score_norm",
        "num_open_accounts",
        "num_delinquencies",
        "interest_rate",
        "loan_amount_x_dti",
        "income_x_credit_score",
    ]

    CATEGORICAL_FEATURES: list[str] = [
        "home_ownership",
        "purpose",
    ]

    TARGET_ENCODED_FEATURES: list[str] = [
        "purpose_target_enc",
        "home_ownership_target_enc",
    ]

    AGGREGATION_FEATURES: list[str] = [
        "avg_days_overdue_30d",
        "avg_days_overdue_90d",
        "avg_days_overdue_180d",
        "max_days_overdue_90d",
        "pct_late_payments_90d",
        "total_paid_90d",
        "payment_consistency_90d",
    ]

    BUREAU_FEATURES: list[str] = [
        "bureau_balance_to_income",
        "inquiries_per_account",
    ]

    ALL_FEATURES: list[str] = (
        NUMERICAL_FEATURES
        + TARGET_ENCODED_FEATURES
        + AGGREGATION_FEATURES
        + BUREAU_FEATURES
    )

    CRITICAL_NO_NULL: list[str] = [
        "loan_to_income",
        "credit_utilization",
        "income_log",
    ]


@lru_cache(maxsize=1)
def get_db_engine():
    from sqlalchemy import create_engine

    return create_engine(
        DBConfig.get_url(),
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
    )
