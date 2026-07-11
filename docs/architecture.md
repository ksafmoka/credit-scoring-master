# Architecture

## System components

```text
┌─────────────────┐      ┌──────────────────────┐      ┌─────────────┐
│ CSV applications │─────▶│     PostgreSQL       │─────▶│   Grafana   │
│ (Lending Club /  │      │ raw / features /     │      │  dashboards │
│  synthetic)      │      │ predictions / mon.   │      └─────────────┘
└─────────────────┘      └──────────┬───────────┘
                                    │
                 ┌──────────────────┼──────────────────┐
                 │                  │                  │
           ┌─────▼─────┐      ┌─────▼─────┐      ┌─────▼─────┐
           │  Airflow  │      │  MLflow   │      │  FastAPI  │
           │   DAGs    │      │  registry │      │  /predict │
           └───────────┘      └───────────┘      └───────────┘
```

## Data flow

1. **`data_ingestion`**  
   CSV → `raw.applications`  
   + synthetic **pre-application** `raw.payment_history`  
   + synthetic `raw.credit_bureau`  
   + validation

2. **`feature_engineering`**  
   numerical / aggregates / target encoding / bureau  
   → staging tables → snapshot `features.application_features`  
   TE maps saved to `artifacts/target_encoding.json`

3. **`model_training`**  
   time split → Optuna (base models) → stacking ensemble  
   → metrics + `mlflow.sklearn.log_model`  
   → local `artifacts/` bundle  
   → register model + alias **`champion`** (if val AUC ≥ threshold)

4. **`batch_prediction`**  
   features + artifact → `predictions.scoring_predictions`

5. **`monitoring`**  
   PSI/KS on features → `monitoring.feature_drift`  
   prediction distribution sanity checks

6. **Online API**  
   load model from MLflow `champion` or local artifact  
   compute online features + TE maps + median fills  
   return PD, risk bucket, top SHAP reasons

## Databases

| DB | Purpose |
|----|---------|
| `credit_scoring` | application data, features, predictions, monitoring |
| `airflow` | Airflow metadata |
| `mlflow` | MLflow backend store |

## Feature groups used by the model

- Numerical / cross features from the application
- Regularized target encoding (`purpose`, `home_ownership`)
- Payment aggregates over 30/90/180d **before** application
- Bureau ratios (`bureau_balance_to_income`, `inquiries_per_account`)

## Serving contract

- Request: application fields (+ optional historical signals)
- Response: `pd_score`, `pd_calibrated`, `risk_bucket`, `top_reasons`, `model_version`
- Missing online aggregates/bureau → training medians (not hard-coded zeros)
