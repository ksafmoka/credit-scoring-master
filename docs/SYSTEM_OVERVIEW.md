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

## Remaining remarks / known limitations

These are **not blockers** for a demo, but good to know:

1. **Synthetic payment & bureau history** — realistic enough for pipeline demos, not real borrower behaviour. Prefer real histories in production.
2. **Optuna cost** — full train DAG with 3 models × 15 trials is heavy; reduce trials for laptops.
3. **Grafana datasource UID** — dashboard panels reference Postgres; on first open you may need to pick the provisioned datasource if UIDs differ.
4. **MLflow stages vs aliases** — code prefers alias `champion`, falls back to stage `Production` for older clients.
5. **Online aggregates** — if the client does not send payment/bureau fields, medians are used (documented train/serve gap for thin-file / pure-online traffic).
6. **No CI workflow file** — `make test` / local pipeline cover unit level; add GitHub Actions when ready.
7. **Airflow LocalExecutor** — fine for demo, not multi-node production HA.
8. **Class imbalance** — no explicit `scale_pos_weight` / undersampling; add if default rate is extreme on real data.
9. **Security** — default passwords in compose are for local demo only.
10. **Repo name** still contains `uplift` historically; product scope is PD-only.

## Health checklist after deploy

- [ ] `make sample-data` produced CSV  
- [ ] `data_ingestion` success  
- [ ] `feature_engineering` success, row count > 0  
- [ ] `model_training` success, artifact files in `artifacts/`  
- [ ] `GET /health` → `model_loaded: true` after `/reload`  
- [ ] `POST /predict` returns PD ∈ [0, 1]  
- [ ] Grafana shows application counts after data load  
