# Credit Scoring System (PD)

End-to-end ML system for **probability of default (PD)** scoring:

CSV → PostgreSQL → Airflow feature engineering → CatBoost / LightGBM / XGBoost + stacking → MLflow → FastAPI → Grafana

> Scope: **default prediction**

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

1. `notebooks/01_EDA.ipynb`
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

Run DAGs in order: `data_ingestion` → `feature_engineering` → `model_training` → `batch_prediction`.

```bash
docker compose exec airflow-scheduler airflow dags trigger data_ingestion
# ensure: SELECT COUNT(*) FROM raw.applications;  > 0
docker compose exec airflow-scheduler airflow dags trigger feature_engineering
docker compose exec airflow-scheduler airflow dags trigger model_training
docker compose exec airflow-scheduler airflow dags trigger batch_prediction
curl -X POST http://localhost:8000/reload
```

After changes under `sql/`: `docker compose down -v && docker compose up -d`.

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
