# Model Card: Credit Scoring (PD) — Dual-Model

## Model details

- **Task:** binary classification — probability of default (PD)
- **Architecture:** dual-model
  - **Model A (with_history):** CatBoost/LightGBM/XGBoost on clients WITH payment history (all 23 features)
  - **Model B (cold_start):** CatBoost/LightGBM/XGBoost on clients WITHOUT payment history (16 features, no aggregation)
- **Selection:** best of 3 boosting models per segment by raw validation AUC
- **Calibration:** isotonic regression — applied only if it improves AUC (conditional)
- **Hyperparameter tuning:** Optuna (TPE, 40 trials) on segment-specific data
- **Explainability:** SHAP TreeExplainer (top reasons in API)

## Training data

- **Source:** Lending Club (2.2M raw applications)
- **Feature subset:** 600k via `features.fe_ids` (300k with payment history + 300k cold-start)
- **Target:** `is_default` (`Charged Off`, `Default`, `Late (31-120 days)`)
- **Split:** time-based (justified by statistical analysis in `notebooks/01_EDA.ipynb`)
  - Train: 2015-01 → 2017-06 (stable regime)
  - Val: 2017-07 → 2017-12
  - Test: 2018-01 → 2018-06

## Metrics

### Model A (with_history)

| Metric | Val |
|--------|-----|
| ROC-AUC | 0.6971 |
| Gini | 0.3942 |
| KS | 0.2795 |
| Brier | 0.22 |

### Model B (cold_start)

| Metric | Val |
|--------|-----|
| ROC-AUC | 0.7126 |
| Gini | 0.4252 |
| KS | ~0.30 |
| Brier | ~0.21 |

## Features

See `FeatureConfig.ALL_FEATURES` (Model A) and `FeatureConfig.COLD_START_FEATURES` (Model B) in `src/config.py` and `docs/feature_documentation.md`.

## Intended use

- Portfolio / educational demonstration of an industrial PD scoring pipeline
- Dual-model serving: separate models for thick-file vs thin-file clients
- Batch and online PD estimates with risk buckets and reason codes
- Demonstrates real-world banking approach: separate scoring for different client segments

**Not intended for** unattended real-world credit decisions without regulatory validation.

## Limitations

- US consumer lending distribution
- Synthetic payment / bureau history used for demo
- Thin-file applicants scored with fewer features (by design)
- No reject inference
- Economic regime shift may degrade performance — monitor PSI/KS

## Ethical considerations

- Protected attributes (race, gender) are not used as model features
- Human review recommended for borderline / high-risk decisions
- Periodic fairness and stability audits recommended
- Dual-model approach ensures thin-file clients are not penalized by missing aggregation features
