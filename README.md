# Credit Scoring System (PD)

End-to-end ML system for **probability of default (PD)** scoring:

CSV → PostgreSQL → Airflow feature engineering → CatBoost / LightGBM / XGBoost + stacking → MLflow → FastAPI → Grafana

> Scope: **default prediction only** (no uplift).

---

## Clean Docker start (recommended)

```bash
cd credit-scoring-uplift-master

# 1) Ensure demo CSV exists
#    (skip if data/sample_applications.csv is already there)
python3.11 scripts/generate_sample_data.py --n 5000   # needs Python 3.11 locally
# or use the file already in data/

# 2) Wipe old volumes + start (SQL 01–04 runs fresh; tables owned by ml_user)
docker compose down -v
docker compose up -d --build

# 3) Wait for airflow-init, then open http://localhost:8080
#    login: admin / admin
```

### Run DAGs (order matters)

1. `data_ingestion`
2. `feature_engineering`
3. `model_training` (longer)
4. `batch_prediction`

```bash
docker compose exec airflow-scheduler airflow dags trigger data_ingestion
```

### API after training

```bash
curl -X POST http://localhost:8000/reload
curl http://localhost:8000/health
```

| Service | URL | Login |
|---------|-----|-------|
| Airflow | http://localhost:8080 | admin / admin |
| MLflow | http://localhost:5000 | — |
| API | http://localhost:8000/docs | — |
| Grafana | http://localhost:3000 | admin / admin |

**Important:** after any change under `sql/`, always use `docker compose down -v` so init scripts re-run. Without `-v` the old volume is reused.

---

## Local (no Docker)

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

## Notebooks

1. `notebooks/01_EDA.ipynb`
2. `notebooks/02_Model_experiments.ipynb` — first cell: `%pip install ...`
3. `notebooks/03_SHAP_analysis.ipynb`

---

## Architecture

```text
CSV (sample / Lending Club)
        │
        ▼
   PostgreSQL  (raw / features / predictions / monitoring)
        │
   Airflow DAGs
     data_ingestion → feature_engineering → model_training
                   → batch_prediction → monitoring
        │
   ┌────┴────┬──────────┐
   MLflow    FastAPI    Grafana
             /predict
```

## Data split

- Target: `is_default` from `loan_status ∈ {Charged Off, Default, Late (31-120 days)}`
- Train ≤ 2022-12-31 · Val ≤ 2023-06-30 · Test > 2023-06-30
- Aggregates use only payments **before** `application_date`

## Docs

- [Apply / run](docs/APPLY_CHANGES.md)
- [System overview](docs/SYSTEM_OVERVIEW.md)
- [Architecture](docs/architecture.md)
- [Features](docs/feature_documentation.md)
- [Model card](docs/model_card.md)

## License

Educational / portfolio project.
