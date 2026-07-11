# Model Card: Credit Scoring (PD)

## Model details

- **Task:** binary classification — probability of default (PD)
- **Models:** CatBoost, LightGBM, XGBoost + stacking ensemble (logistic meta-model)
- **Calibration:** isotonic regression on validation (`CalibratedClassifierCV`, prefit)
- **Hyperparameter tuning:** Optuna (TPE)
- **Explainability:** SHAP TreeExplainer (top reasons in API)

## Training data

- **Source:** Lending Club-compatible applications or synthetic sample
- **Target:** `is_default` (`Charged Off`, `Default`, `Late (31-120 days)`)
- **Split:** time-based  
  - Train ≤ 2022-12-31  
  - Val ≤ 2023-06-30  
  - Test > 2023-06-30

## Metrics

Fill after a training run (MLflow / model card JSON):

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| ROC-AUC | | | |
| PR-AUC | | | |
| KS | | | |
| Gini | | | |
| Brier | | | |

## Features

See `FeatureConfig.ALL_FEATURES` in `src/config.py` and `docs/feature_documentation.md`.

## Intended use

- Portfolio / educational demonstration of an industrial PD scoring pipeline
- Batch and online PD estimates with risk buckets and reason codes

**Not intended for** unattended real-world credit decisions without regulatory validation.

## Limitations

- US consumer lending distribution (or synthetic proxy)
- Synthetic payment / bureau history used when real history is unavailable
- Thin-file applicants may have sparse aggregates (median imputation online)
- No reject inference
- Economic regime shift may degrade performance — monitor PSI/KS

## Ethical considerations

- Protected attributes (race, gender) are not used as model features
- Human review recommended for borderline / high-risk decisions
- Periodic fairness and stability audits recommended
