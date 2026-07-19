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
    # Overridable via env (set by prepare_lending_club.py → .env)
    TRAIN_END_DATE: str = os.getenv("TRAIN_END_DATE", "2022-12-31")
    VAL_END_DATE: str = os.getenv("VAL_END_DATE", "2023-06-30")
    RANDOM_SEED: int = 42
    N_OPTUNA_TRIALS: int = int(os.getenv("N_OPTUNA_TRIALS", "15"))
    CV_FOLDS: int = 5
    TARGET_COL: str = "is_default"
    ID_COL: str = "application_id"
    DATE_COL: str = "application_date"
    MIN_AUC_FOR_REGISTRATION: float = float(
        os.getenv("MIN_AUC_FOR_REGISTRATION", "0.70")
    )
    CALIBRATION_METHOD: str = os.getenv(
        "CALIBRATION_METHOD", "isotonic"
    )  # isotonic | sigmoid | none
    # If train/val empty with fixed dates, auto-split by quantiles of application_date
    AUTO_DATE_SPLIT: bool = os.getenv("AUTO_DATE_SPLIT", "true").lower() in {
        "1",
        "true",
        "yes",
    }


class IngestionConfig:
    """
    Synthetic history generation controls.

    Why 300k default (was 150k): 2.2M * 12 = 27M payment rows OOMs Airflow worker.
    Batched + random sampling gives uniform train coverage and keeps memory <500MB.
    300k payments + 400k FE = 75% coverage with history, 25% cold-start (realistic).
    Env-overridable.
    """

    # None = all apps; int = limit applications
    PAYMENT_HISTORY_MAX_APPS: int | None = (
        int(v) if (v := os.getenv("PAYMENT_HISTORY_MAX_APPS", "300000").strip()) != "" else None
    )
    PAYMENT_HISTORY_BATCH_SIZE: int = int(
        os.getenv("PAYMENT_HISTORY_BATCH_SIZE", "5000")
    )
    PAYMENT_HISTORY_N_PER_LOAN: int = int(
        os.getenv("PAYMENT_HISTORY_N_PER_LOAN", "12")
    )
    BUREAU_MAX_APPS: int | None = (
        int(v) if (v := os.getenv("BUREAU_MAX_APPS", "").strip()) != "" else None
    )
    BUREAU_BATCH_SIZE: int = int(os.getenv("BUREAU_BATCH_SIZE", "10000"))
    SAMPLE_STRATEGY: str = os.getenv("HISTORY_SAMPLE_STRATEGY", "random").strip().lower()


class MonitoringConfig:
    PSI_THRESHOLD: float = 0.15
    PSI_WARNING: float = 0.10
    AUC_DROP_THRESHOLD: float = 0.03
    MIN_PREDICTIONS_FOR_CHECK: int = 100

    # Read at call-time so Docker env changes apply without import-order issues
    @staticmethod
    def telegram_bot_token() -> str:
        return os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

    @staticmethod
    def telegram_chat_id() -> str:
        return os.getenv("TELEGRAM_CHAT_ID", "").strip()


class FeatureEngineeringConfig:
    """
    Feature engineering batching + cold-start aware sampling.

    Old: 2.2M RAM + single to_sql transaction (4min+) -> DNS/heartbeat fail.
    New: 100k chunks, 50k commit batches, sequential DAG, consistent fe_ids table
    with 75% history coverage (300k with payments + 100k cold-start = 400k total).
    """

    NUMERICAL_BATCH_SIZE: int = int(
        os.getenv("FE_NUMERICAL_BATCH_SIZE", "100000")
    )
    BUREAU_BATCH_SIZE: int = int(os.getenv("FE_BUREAU_BATCH_SIZE", "100000"))
    AGGREGATION_BATCH_SIZE: int = int(
        os.getenv("FE_AGGREGATION_BATCH_SIZE", "100000")
    )
    TARGET_ENCODING_BATCH_SIZE: int = int(
        os.getenv("FE_TARGET_ENCODING_BATCH_SIZE", "100000")
    )
    STAGING_WRITE_BATCH_SIZE: int = int(
        os.getenv("FE_STAGING_WRITE_BATCH_SIZE", "50000")
    )
    # None = all 2.2M for full prod, 400k = 300k with history + 100k cold-start for weak PC
    MAX_APPS: int | None = (
        int(v) if (v := os.getenv("FE_MAX_APPS", "").strip()) != "" else None
    )


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

    # Features available WITHOUT payment history (for cold-start model)
    COLD_START_FEATURES: list[str] = (
        NUMERICAL_FEATURES
        + TARGET_ENCODED_FEATURES
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
