# Credit Scoring System

## Overview
End-to-end ML system for credit default prediction (PD scoring)
with Airflow orchestration, MLflow experiment tracking,
FastAPI serving, and Grafana monitoring.

## Architecture
```text
Lending Club CSV
    → PostgreSQL (raw schema)
    → Feature Engineering (Airflow DAG)
    → Model Training: CatBoost / LightGBM / XGBoost + Stacking (Airflow DAG)
    → MLflow Model Registry
    → FastAPI Scoring API
    → Grafana Monitoring Dashboard
    
# Stack
Orchestration: Apache Airflow 2.8
Experiment Tracking: MLflow 2.10
Database: PostgreSQL 15
Serving: FastAPI + Uvicorn
Monitoring: Grafana + custom PSI/KS drift checks
Models: CatBoost, LightGBM, XGBoost, Stacking Ensemble
Hyperparameter Tuning: Optuna
Explainability: SHAP
# Quick Start

# 1. Start all services
make up

# 2. Wait for postgres healthcheck, then init DB schemas
make init-db

# 3. Access services
# Airflow:  http://localhost:8080  (admin/admin)
# MLflow:   http://localhost:5000
# API:      http://localhost:8000/docs
# Grafana:  http://localhost:3000  (admin/admin)

# 4. Trigger data ingestion DAG from Airflow UI
# 5. Trigger feature engineering DAG
# 6. Trigger model training DAG

Pipelines
DAG	Schedule	Description
data_ingestion	@daily	Load raw data, generate payment history, validate
feature_engineering	@daily	Compute numerical, aggregation, target-encoded features
model_training	manual trigger	Train 3 models + ensemble, hyperopt, leakage check, register
monitoring	@daily	Feature drift (PSI/KS), prediction distribution checks

Tests
make test

Data
Source: Lending Club (Kaggle)

Target: is_default — binary flag based on loan_status ∈ {Charged Off, Default, Late (31-120 days)}

Split: time-based. Train ≤ 2022-12-31, Val ≤ 2023-06-30, Test > 2023-06-30.

---

## 2. `.env.example`

```env
# .env.example

LENDING_CLUB_CSV_PATH=/opt/airflow/data/lending_club.csv

# PostgreSQL (application database)
APP_DB_HOST=postgres
APP_DB_PORT=5432
APP_DB_NAME=credit_scoring
APP_DB_USER=ml_user
APP_DB_PASSWORD=ml_password

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000

# Telegram alerts (optional)
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=