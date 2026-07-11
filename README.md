# Credit Scoring System (PD)

End-to-end ML system for **probability of default (PD)** scoring:

CSV → PostgreSQL → Airflow feature engineering → CatBoost / LightGBM / XGBoost + stacking → MLflow → FastAPI → Grafana

> Scope: **default prediction**

## Quick start (local, no Docker)

```bash
# 1. Create venv
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS / Linux:
# source .venv/bin/activate

# 2. Install deps
pip install -r requirements/requirements-local.txt

# 3. Smoke: sample data + train + tests
# Windows PowerShell:
$env:PYTHONPATH="."
python scripts/generate_sample_data.py --n 3000
python scripts/run_local_pipeline.py
pytest tests/ -q

# macOS / Linux:
# PYTHONPATH=. make smoke
```

Artifacts appear in `artifacts/` (`model.pkl`, `artifact_meta.json`, `target_encoding.json`).

## Notebooks

Open in VS Code / Jupyter from the **project root** (or `notebooks/`).

1. `notebooks/01_EDA.ipynb`
2. `notebooks/02_Model_experiments.ipynb`
3. `notebooks/03_SHAP_analysis.ipynb`

**First cell** installs packages with `%pip install ...` into the active kernel.  
Run top-to-bottom. If imports fail after install → **Kernel → Restart** → run again.

## Docker full stack

```bash
cp .env.example .env
# ensure sample data exists
python scripts/generate_sample_data.py --n 5000

make up
# wait ~1–2 min for postgres + airflow-init

# Trigger pipelines
docker compose exec airflow-scheduler airflow dags trigger data_ingestion
docker compose exec airflow-scheduler airflow dags trigger feature_engineering
docker compose exec airflow-scheduler airflow dags trigger model_training
docker compose exec airflow-scheduler airflow dags trigger batch_prediction

curl -X POST http://localhost:8000/reload
curl http://localhost:8000/health
```

| Service | URL | Login |
|---------|-----|-------|
| Airflow | http://localhost:8080 | admin / admin |
| MLflow | http://localhost:5000 | — |
| API docs | http://localhost:8000/docs | — |
| Grafana | http://localhost:3000 | admin / admin |

If Postgres volume is old after SQL changes:

```bash
docker compose down -v
make up
```

## Architecture

<img width="980" height="640" alt="architecture_diagram" src="https://github.com/user-attachments/assets/e1ec2801-5ce1-4fe2-8f61-2ba8af37f38f" />
<svg xmlns="http://www.w3.org/2000/svg" width="980" height="640" viewBox="0 0 980 640">


## DAGs

| DAG | Schedule | Description |
|-----|----------|-------------|
| `data_ingestion` | @daily | Load CSV + synthetic pre-app payments/bureau |
| `feature_engineering` | @daily | Numerical / agg / TE / bureau snapshot |
| `model_training` | manual | Optuna + 3 boosters + stacking + register |
| `batch_prediction` | @daily | Batch PD scores |
| `monitoring` | @daily | PSI/KS + prediction health |

## Data split

- Target: `is_default` from `loan_status ∈ {Charged Off, Default, Late (31-120 days)}`
- Train ≤ 2022-12-31 · Val ≤ 2023-06-30 · Test > 2023-06-30  
- Aggregates use only payments **before** `application_date`

## Project layout

```text
configs/   dags/   docker/   docs/   monitoring/
notebooks/ requirements/ scripts/ sql/ src/ tests/
data/sample_applications.csv   # demo CSV (regenerate via scripts/)
```

## Docs

- [Apply / run](docs/APPLY_CHANGES.md)
- [System overview](docs/SYSTEM_OVERVIEW.md)
- [Architecture](docs/architecture.md)
- [Features](docs/feature_documentation.md)
- [Model card](docs/model_card.md)

## License

Educational / portfolio project. Not a production credit decisioning system without further validation.
