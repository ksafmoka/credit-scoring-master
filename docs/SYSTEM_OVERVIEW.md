# System overview

## How the system works

```text
                         ┌──────────────────────────────────────────────┐
                         │                 DATA LAYER                   │
                         │  data/*.csv → raw.applications (2.2M)        │
                         │              raw.payment_history (300k apps) │
                         │              raw.credit_bureau (600k)        │
                         └──────────────────────┬───────────────────────┘
                                                │
                         ┌──────────────────────▼───────────────────────┐
                         │            FEATURE ENGINEERING               │
                         │  fe_ids: 600k (300k history + 300k cold)    │
                         │  numerical │ aggregates │ TE │ bureau        │
                         │  → features.application_features (snapshot)  │
                         └──────────────────────┬───────────────────────┘
                                                │
              ┌─────────────────────────────────┼─────────────────────────┐
              │                                 │                         │
              ▼                                 ▼                         ▼
   ┌────────────────────┐          ┌────────────────────┐     ┌────────────────────┐
   │  MODEL TRAINING    │          │  BATCH PREDICTION  │     │  MONITORING        │
   │  Dual-model:       │          │  Dual-model:       │     │  PSI / KS drift    │
   │  with_history      │          │  score by segment  │     │  PD distribution  │
   │  cold_start        │          │  → predictions.*   │     │  Telegram alerts  │
   │  Optuna + calib    │          └────────────────────┘     └────────────────────┘
   │  MLflow 2 models   │
   └──────────┬─────────┘
              │
              ▼
   ┌────────────────────┐
   │  FASTAPI SERVING   │
   │  Dual-model routing│
   │  /health /predict  │
   │  /reload /model-info│
   │  PD + risk + SHAP  │
   └────────────────────┘
```

### Online scoring path (dual-model)

```text
ScoringRequest
   → check: are payment history fields provided?
   → if yes: load with_history model (23 features)
   → if no:  load cold_start model (16 features)
   → compute application numerics + TE maps + optional features
   → model.predict_proba
   → risk bucket + SHAP top reasons
   → ScoringResponse (model_version shows which model was used)
```

### Offline training path (dual-model)

```text
Load features + payment_history IDs
   → split: with_history (300k) / cold_start (300k)
   → for each segment:
       Optuna on SEGMENT data (40 trials)
       train catboost + lightgbm + xgboost
       pick best by raw val AUC
       conditional calibration (only if AUC improves)
       save artifact
   → register both models in MLflow
```

## Component map

| Component | Entry point |
|-----------|-------------|
| Config | `src/config.py` |
| Ingestion | `src/data/ingestion.py` + `dags/dag_data_ingestion.py` |
| Features | `src/features/*` + `dags/dag_feature_engineering.py` |
| Train | `src/models/scoring/*` + `dags/dag_training.py` |
| Batch score | `dags/dag_batch_prediction.py` |
| API | `src/serving/app.py` + `src/serving/predict.py` |
| Drift | `src/monitoring/data_drift.py` + `dags/dag_monitoring.py` |
| Learning curve | `scripts/learning_curve.py` |
| EDA | `notebooks/01_EDA.ipynb` |
| Infra | `docker-compose.yml`, `sql/*`, `Makefile` |

## Health checklist after deploy

- [ ] `data_ingestion` success (raw.applications > 0)
- [ ] `feature_engineering` success (600k rows, ~75% with payment history)
- [ ] `model_training` success (both models registered in MLflow)
- [ ] `batch_prediction` success (predictions table populated)
- [ ] `GET /health` → `history_model_loaded: true, cold_start_model_loaded: true`
- [ ] `POST /predict` with history fields → model_version shows `with_history`
- [ ] `POST /predict` without history → model_version shows `cold_start`
- [ ] `monitoring` DAG runs without errors
- [ ] Grafana shows prediction distribution
- [ ] Learning curve saved in `artifacts/`
