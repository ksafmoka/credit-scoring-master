# Feature documentation

## Feature groups

### Numerical & cross (`src/features/numerical.py`) — used by both models

| Feature | Formula / source |
|---------|------------------|
| `loan_to_income` | `loan_amount / income` |
| `credit_utilization` | `loan_amount / total_credit_limit` |
| `income_log` | `log1p(income)` |
| `loan_amount_log` | `log1p(loan_amount)` |
| `dti_ratio_clipped` | `clip(dti_ratio, 0, 100)` |
| `employment_years` | employment tenure |
| `credit_score_norm` | `(credit_score - 300) / 550` clipped to [0, 1] |
| `num_open_accounts` | open trade lines |
| `num_delinquencies` | delinquency count |
| `interest_rate` | loan APR |
| `loan_amount_x_dti` | product cross |
| `income_x_credit_score` | product cross |

### Aggregation (`src/features/aggregations.py`) — Model A (with_history) only

Only payments with `payment_date < application_date`.

| Feature | Window |
|---------|--------|
| `avg_days_overdue_{30,90,180}d` | mean overdue days |
| `max_days_overdue_90d` | max overdue |
| `pct_late_payments_90d` | share of late payments |
| `total_paid_90d` | sum paid |
| `payment_consistency_90d` | `1 - std(days_overdue)/90` clipped |

### Target encoding (`src/features/target_encoding.py`) — used by both models

| Feature | Notes |
|---------|-------|
| `purpose_target_enc` | fit on train period only, smoothing=20 |
| `home_ownership_target_enc` | same |

Maps are persisted to `artifacts/target_encoding.json` for online parity.
**No noise** on inference transforms.

### Bureau (`src/features/bureau.py`) — used by both models

Latest bureau report **before** application date:

| Feature | Formula |
|---------|---------|
| `bureau_balance_to_income` | `total_balance / income` |
| `inquiries_per_account` | `num_inquiries_6m / num_active_loans` |

## Feature sets per model

| Feature set | Config | Model A | Model B |
|-------------|--------|---------|---------|
| Numerical (12) | `FeatureConfig.NUMERICAL_FEATURES` | ✅ | ✅ |
| Target encoding (2) | `FeatureConfig.TARGET_ENCODED_FEATURES` | ✅ | ✅ |
| Aggregation (7) | `FeatureConfig.AGGREGATION_FEATURES` | ✅ | ❌ |
| Bureau (2) | `FeatureConfig.BUREAU_FEATURES` | ✅ | ✅ |
| **Total** | `ALL_FEATURES` / `COLD_START_FEATURES` | **23** | **16** |

## Leakage protection

- Aggregates & bureau use only pre-application events
- Target encoding fitted on train cutoff only
- Time-based train/val/test split (justified by EDA statistical tests)
- Leakage DAG checks: train/test ID overlap + future payments
- fe_ids consistency ensures all FE tasks operate on the same application set
