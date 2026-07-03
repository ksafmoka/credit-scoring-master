```markdown
# Model Card: Credit Scoring (PD)

## Model Details
- **Task**: Binary classification (probability of default)
- **Models**: CatBoost, LightGBM, XGBoost + Stacking Ensemble
- **Calibration**: Isotonic regression (planned)
- **Hyperparameter tuning**: Optuna (50 trials)

## Training Data
- **Source**: Lending Club
- **Target**: is_default (Charged Off, Default, Late 31-120 days)
- **Split**: Time-based. Train ≤ 2022-12-31, Val ≤ 2023-06-30, Test > 2023-06-30

## Metrics (to be filled after training)
| Metric | Train | Val | Test |
|--------|-------|-----|------|
| ROC-AUC | | | |
| PR-AUC | | | |
| KS | | | |
| Gini | | | |
| Brier | | | |

## Features
18 features from 4 groups: numerical, target-encoded, aggregation, cross-features.
Full list in `src/config.py :: FeatureConfig.ALL_FEATURES`.

## Limitations
- Trained on US lending data only
- Synthetic payment history (not real borrower behavior)
- Treatment flag is randomly generated (not real experiment)
- No reject inference
- Aggregation features unavailable in online serving (filled with zeros)

## Ethical Considerations
- Protected attributes (race, gender, age) not used as features
- SHAP explanations provided for each prediction