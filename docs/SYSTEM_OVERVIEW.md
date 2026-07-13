# System overview (current)

## How the system works now

```text
                         ┌──────────────────────────────────────────────┐
                         │                 DATA LAYER                   │
                         │  data/*.csv  →  raw.applications             │
                         │               raw.payment_history (pre-app)  │
                         │               raw.credit_bureau              │
                         └──────────────────────┬───────────────────────┘
                                                │
                         ┌──────────────────────▼───────────────────────┐
                         │            FEATURE ENGINEERING               │
                         │  numerical │ aggregates │ TE │ bureau        │
                         │  → features.application_features (snapshot)  │
                         │  → artifacts/target_encoding.json            │
                         └──────────────────────┬───────────────────────┘
                                                │
              ┌─────────────────────────────────┼─────────────────────────┐
              │                                 │                         │
              ▼                                 ▼                         ▼
   ┌────────────────────┐          ┌────────────────────┐     ┌────────────────────┐
   │  MODEL TRAINING    │          │  BATCH PREDICTION  │     │  MONITORING        │
   │  CatBoost/LGBM/XGB │          │  score snapshot    │     │  PSI / KS drift    │
   │  Optuna + stack    │          │  → predictions.*   │     │  PD distribution  │
   │  calibrate isotonic│          └────────────────────┘     └────────────────────┘
   │  MLflow log+register│
   │  artifacts/ bundle │
   └──────────┬─────────┘
              │
              ▼
   ┌────────────────────┐
   │  FASTAPI SERVING   │
   │  /health /predict  │
   │  /reload /model-info│
   │  PD + risk + SHAP  │
   └────────────────────┘
```

### Online scoring path

```text
ScoringRequest
   → compute application numerics
   → apply TE maps from artifact (not hard-coded 0.15)
   → fill missing history/bureau with training medians
   → model.predict_proba
   → risk bucket + SHAP top reasons
   → ScoringResponse
```

### Offline training path

```text
time split (train/val/test)
   → median impute
   → Optuna on val AUC (data loaded once)
   → refit best params + isotonic calibration
   → stacking ensemble (TimeSeriesSplit OOF)
   → leakage checks
   → log model + bundle; alias champion if AUC ≥ threshold
```

## Component map

| Component | Entry point |
|-----------|-------------|
| Config | `src/config.py` |
| Ingestion | `src/data/ingestion.py` + `dags/dag_data_ingestion.py` |
| Features | `src/features/*` + `dags/dag_feature_engineering.py` |
| Train | `src/models/scoring/*` + `dags/dag_training.py` |
| Batch score | `dags/dag_batch_prediction.py` |
| API | `src/serving/app.py` |
| Drift | `src/monitoring/data_drift.py` + `dags/dag_monitoring.py` |
| Infra | `docker-compose.yml`, `sql/*`, `Makefile` |


## Health checklist after deploy

- [ ] `make sample-data` produced CSV  
- [ ] `data_ingestion` success  
- [ ] `feature_engineering` success, row count > 0  
- [ ] `model_training` success, artifact files in `artifacts/`  
- [ ] `GET /health` → `model_loaded: true` after `/reload`  
- [ ] `POST /predict` returns PD ∈ [0, 1]  
- [ ] Grafana shows application counts after data load  
