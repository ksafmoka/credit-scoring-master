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
           ┌─────▼─────┐      ┌─────▼─────┐      ┌─────▼──────────┐
           │  Airflow  │      │  MLflow   │      │    FastAPI     │
           │   DAGs    │      │  2 models │      │  dual-model    │
           └───────────┘      └───────────┘      │  routing       │
                                                  └────────────────┘
```

## Data architecture

- **Raw data lake**: 2.2M applications always kept in `raw.applications`
- **Feature sampling**: 600k via `features.fe_ids` (300k with payment history + 300k cold-start)
- **Payment history**: 300k applications × 12 pre-application payments
- **Cold-start coverage**: 50% of features have payment history, 50% don't

### fe_ids consistency

All feature engineering tasks (numerical, aggregation, target encoding, bureau) use `features.fe_ids` as the single source of truth. This ensures application_ids match across all staging tables before the final merge.

## Data flow

1. **`data_ingestion`**
   CSV → `raw.applications`
   + synthetic **pre-application** `raw.payment_history` (300k apps)
   + synthetic `raw.credit_bureau`
   + validation

2. **`feature_engineering`**
   - `select_fe_ids`: creates consistent 600k ID set (300k with history + 300k cold-start)
   - All FE tasks filter by `fe_ids` → no mismatched IDs in merge
   - numerical / aggregates / target encoding / bureau
   - staging tables → snapshot `features.application_features`
   - TE maps saved to `artifacts/target_encoding.json`

3. **`model_training`** (dual-model)
   - Splits features by `payment_history` membership
   - **with_history segment**: ~300k clients, ALL features (23), Optuna on segment data
   - **cold_start segment**: ~300k clients, non-aggregation features only (16)
   - Each segment: train CatBoost + LightGBM + XGBoost → pick best by raw AUC
   - Conditional calibration: applied only if it improves AUC
   - Register both models in MLflow: `credit_scoring_with_history`, `credit_scoring_cold_start`

4. **`batch_prediction`** (dual-model)
   - Splits features by payment_history membership
   - Scores each segment with its model
   - → `predictions.scoring_predictions`

5. **`monitoring`**
   PSI/KS on features → `monitoring.feature_drift`
   prediction distribution sanity checks
   Telegram alerts on drift

6. **Online API** (dual-model routing)
   - Request arrives with optional payment history fields
   - If history fields present → `credit_scoring_with_history` model
   - If no history → `credit_scoring_cold_start` model
   - Returns PD, risk bucket, top SHAP reasons, model_version

## Databases

| DB | Purpose |
|----|---------|
| `credit_scoring` | application data, features, predictions, monitoring |
| `airflow` | Airflow metadata |
| `mlflow` | MLflow backend store |

## Time-window selection

EDA notebook (`notebooks/01_EDA.ipynb`) statistically proves distribution shift:
- Chi-squared test: default rates differ significantly across eras (p < 0.001)
- KS test: credit_score, income, DTI distributions differ between pre-2015 and 2015+
- PSI: >0.25 (significant shift) between crisis era and stable era
- Coefficient of variation within 2015–2018 window < 0.3 → stable regime

**Training window**: 2015-01 to 2017-06 (train), 2017-07 to 2017-12 (val), 2018-01 to 2018-06 (test)

## Feature groups

| Group | Model A (history) | Model B (cold-start) |
|-------|-------------------|---------------------|
| Numerical / cross (12) | ✅ | ✅ |
| Target encoding (2) | ✅ | ✅ |
| Aggregation (7) | ✅ | ❌ |
| Bureau (2) | ✅ | ✅ |
| **Total** | **23** | **16** |

## Serving contract

- Request: application fields (+ optional historical signals)
- Response: `pd_score`, `pd_calibrated`, `risk_bucket`, `top_reasons`, `model_version`
- Missing online aggregates/bureau → training medians (not hard-coded zeros)
- `model_version` indicates which segment model was used
