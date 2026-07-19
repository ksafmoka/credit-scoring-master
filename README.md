# Credit Scoring System (PD)

End-to-end ML system for **probability of default (PD)** scoring with **dual-model architecture**:

CSV → PostgreSQL → Airflow feature engineering → CatBoost / LightGBM / XGBoost → MLflow → FastAPI (dual-model routing) → Grafana

> **Scope:** default prediction with separate models for thick-file (with payment history) and thin-file (cold-start) clients.

## Architecture highlights

- **Dual-model serving**: API routes requests to the appropriate model based on available payment history
- **fe_ids consistency**: all feature engineering tasks use the same `features.fe_ids` table → no mismatched application_ids
- **Time-window analysis**: EDA with statistical tests (chi-squared, KS, PSI) proves distribution shift across eras → training on 2015–2018 stable regime
- **Learning curve**: data-driven decision on optimal training size per segment
- **Conditional calibration**: isotonic calibration applied only when it improves AUC

## Data

- Prefer `data/lending_club.csv` (auto-selected if present).
- Fallback: `data/sample_applications.csv`.

```bash
# optional sample
python scripts/generate_sample_data.py --n 5000
```

## Quick start (local, no Docker)

**Python 3.10 / 3.11 / 3.12 only** (not 3.13/3.14).

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip setuptools wheel
pip install -r requirements/requirements-local.txt
export PYTHONPATH=.                # PowerShell: $env:PYTHONPATH="."
python scripts/generate_sample_data.py --n 3000
python scripts/run_local_pipeline.py
pytest tests/ -q
```

Artifacts appear in `artifacts/`.

## Notebooks

1. `notebooks/01_EDA.ipynb` — temporal distribution analysis, regime detection, statistical tests
2. `notebooks/02_Model_experiments.ipynb` — first cell `%pip install ...`
3. `notebooks/03_SHAP_analysis.ipynb`

## Docker

```bash
cp .env.example .env
docker compose up -d
# If images are missing and PyPI is stable: make rebuild

# Airflow http://localhost:8080  (admin/admin)
# MLflow  http://localhost:5000
# API     http://localhost:8000/docs
# Grafana http://localhost:3000  (admin/admin)
```

Run DAGs in order:
```bash
docker compose exec airflow-scheduler airflow dags trigger data_ingestion
docker compose exec airflow-scheduler airflow dags trigger feature_engineering
docker compose exec airflow-scheduler airflow dags trigger model_training
docker compose exec airflow-scheduler airflow dags trigger batch_prediction
docker compose exec airflow-scheduler airflow dags trigger monitoring
curl -X POST http://localhost:8000/reload
```

After changes under `sql/`: `docker compose down -v && docker compose up -d`.

## Dual-model serving

The API automatically routes requests based on available payment history:

```bash
# With payment history → with_history model
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "application_id": 1,
    "loan_amount": 15000, "income": 60000, "loan_term": 36,
    "interest_rate": 12.5, "employment_years": 5, "credit_score": 700,
    "dti_ratio": 20, "num_open_accounts": 5, "num_delinquencies": 0,
    "total_credit_limit": 50000, "home_ownership": "RENT",
    "purpose": "debt_consolidation",
    "avg_days_overdue_90d": 5.2, "pct_late_payments_90d": 0.15
  }'

# Without payment history → cold_start model
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "application_id": 2,
    "loan_amount": 10000, "income": 45000, "loan_term": 36,
    "interest_rate": 15.0, "employment_years": 2, "credit_score": 650,
    "dti_ratio": 30, "num_open_accounts": 3, "num_delinquencies": 1,
    "total_credit_limit": 25000, "home_ownership": "RENT",
    "purpose": "credit_card"
  }'
```

## Learning curve

```bash
docker compose exec airflow-scheduler python scripts/learning_curve.py
```

Produces `artifacts/learning_curve.csv` and `artifacts/learning_curve.png` with separate curves for each segment.

## Feature drift → Telegram

Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`.
If Docker cannot reach `api.telegram.org`, run on the host:

```bash
make telegram-loop
```

## Docs

- [Lending Club](docs/LENDING_CLUB.md)
- [Architecture](docs/architecture.md)
- [Features](docs/feature_documentation.md)
- [Model card](docs/model_card.md)
- [System overview](docs/SYSTEM_OVERVIEW.md)

## License

Educational / portfolio project.
