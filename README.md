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
<img width="980" height="640" alt="architecture_diagram" src="https://github.com/user-attachments/assets/e1ec2801-5ce1-4fe2-8f61-2ba8af37f38f" />
<svg xmlns="http://www.w3.org/2000/svg" width="980" height="640" viewBox="0 0 980 640">
  <defs>
    <linearGradient id="g1" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#1f6feb"/>
      <stop offset="100%" stop-color="#388bfd"/>
    </linearGradient>
    <filter id="s" x="-5%" y="-5%" width="110%" height="110%">
      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-opacity="0.2"/>
    </filter>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 0 L10 5 L0 10 z" fill="#57606a"/>
    </marker>
  </defs>
  <rect width="980" height="640" fill="#f6f8fa"/>
  <text x="40" y="40" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="22" font-weight="700" fill="#24292f">Credit Scoring PD — system overview</text>
  <text x="40" y="64" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="13" fill="#57606a">Default prediction only · Airflow · MLflow · FastAPI · Grafana</text>

  <!-- CSV -->
  <rect x="40" y="100" width="160" height="70" rx="12" fill="#fff" stroke="#d0d7de" filter="url(#s)"/>
  <text x="120" y="130" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="14" font-weight="600" fill="#24292f">CSV source</text>
  <text x="120" y="150" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="11" fill="#57606a">Lending Club / sample</text>

  <!-- Postgres -->
  <rect x="280" y="90" width="220" height="100" rx="12" fill="#fff" stroke="#1f6feb" stroke-width="2" filter="url(#s)"/>
  <text x="390" y="120" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="15" font-weight="700" fill="#1f6feb">PostgreSQL</text>
  <text x="390" y="142" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="11" fill="#57606a">raw · features</text>
  <text x="390" y="160" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="11" fill="#57606a">predictions · monitoring</text>

  <!-- Airflow -->
  <rect x="580" y="90" width="340" height="100" rx="12" fill="#fff" stroke="#8250df" stroke-width="2" filter="url(#s)"/>
  <text x="750" y="118" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="15" font-weight="700" fill="#8250df">Apache Airflow</text>
  <text x="750" y="140" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="11" fill="#57606a">ingestion → features → training</text>
  <text x="750" y="158" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="11" fill="#57606a">batch prediction · monitoring</text>

  <!-- arrows top -->
  <line x1="200" y1="135" x2="275" y2="135" stroke="#57606a" stroke-width="2" marker-end="url(#arrow)"/>
  <line x1="500" y1="140" x2="575" y2="140" stroke="#57606a" stroke-width="2" marker-end="url(#arrow)"/>
  <path d="M750 190 V230" stroke="#57606a" stroke-width="2" marker-end="url(#arrow)"/>
  <path d="M390 190 V230" stroke="#57606a" stroke-width="2" marker-end="url(#arrow)"/>

  <!-- Feature box -->
  <rect x="40" y="240" width="420" height="120" rx="12" fill="#fff" stroke="#bf8700" stroke-width="2" filter="url(#s)"/>
  <text x="250" y="270" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="15" font-weight="700" fill="#bf8700">Feature engineering</text>
  <text x="250" y="295" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="12" fill="#57606a">numerical · payment aggs (pre-app) · bureau · TE</text>
  <text x="250" y="315" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="12" fill="#57606a">time-based snapshot → application_features</text>
  <text x="250" y="335" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="12" fill="#57606a">artifacts/target_encoding.json</text>

  <!-- Models -->
  <rect x="500" y="240" width="420" height="120" rx="12" fill="#fff" stroke="#1a7f37" stroke-width="2" filter="url(#s)"/>
  <text x="710" y="270" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="15" font-weight="700" fill="#1a7f37">PD models</text>
  <text x="710" y="295" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="12" fill="#57606a">CatBoost / LightGBM / XGBoost + stacking</text>
  <text x="710" y="315" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="12" fill="#57606a">Optuna · isotonic calibration · SHAP</text>
  <text x="710" y="335" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="12" fill="#57606a">MLflow registry alias: champion</text>

  <line x1="460" y1="300" x2="495" y2="300" stroke="#57606a" stroke-width="2" marker-end="url(#arrow)"/>

  <!-- Serving -->
  <rect x="40" y="420" width="280" height="110" rx="12" fill="url(#g1)" filter="url(#s)"/>
  <text x="180" y="455" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="15" font-weight="700" fill="#fff">FastAPI serving</text>
  <text x="180" y="480" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="12" fill="#e6f0ff">/predict PD + risk bucket</text>
  <text x="180" y="500" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="12" fill="#e6f0ff">/health · /reload · SHAP reasons</text>

  <!-- Batch -->
  <rect x="350" y="420" width="280" height="110" rx="12" fill="#fff" stroke="#cf222e" stroke-width="2" filter="url(#s)"/>
  <text x="490" y="455" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="15" font-weight="700" fill="#cf222e">Batch prediction</text>
  <text x="490" y="480" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="12" fill="#57606a">score feature snapshot</text>
  <text x="490" y="500" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="12" fill="#57606a">→ predictions.scoring_predictions</text>

  <!-- Monitoring -->
  <rect x="660" y="420" width="260" height="110" rx="12" fill="#fff" stroke="#0969da" stroke-width="2" filter="url(#s)"/>
  <text x="790" y="455" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="15" font-weight="700" fill="#0969da">Monitoring</text>
  <text x="790" y="480" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="12" fill="#57606a">PSI / KS drift · PD health</text>
  <text x="790" y="500" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="12" fill="#57606a">Grafana dashboards</text>

  <path d="M710 360 V400" stroke="#57606a" stroke-width="2" marker-end="url(#arrow)"/>
  <path d="M180 360 V415" stroke="#57606a" stroke-width="2" marker-end="url(#arrow)" transform="translate(530,0)"/>
  <line x1="710" y1="380" x2="180" y2="415" stroke="#57606a" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#arrow)"/>
  <line x1="710" y1="380" x2="490" y2="415" stroke="#57606a" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#arrow)"/>
  <line x1="710" y1="380" x2="790" y2="415" stroke="#57606a" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#arrow)"/>

  <text x="40" y="580" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="12" fill="#57606a">Leakage guards: pre-application payments/bureau · TE fit on train only · time-based split · DAG leakage checks</text>
  <text x="40" y="602" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="12" fill="#57606a">Serving parity: model + feature medians + TE maps packaged in artifacts/</text>
</svg>


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
