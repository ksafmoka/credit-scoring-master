# Feature documentation

## Feature groups

### Numerical & cross (`src/features/numerical.py`)

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

### Aggregation (`src/features/aggregations.py`)

Only payments with `payment_date < application_date`.

| Feature | Window |
|---------|--------|
| `avg_days_overdue_{30,90,180}d` | mean overdue days |
| `max_days_overdue_90d` | max overdue |
| `pct_late_payments_90d` | share of late payments |
| `total_paid_90d` | sum paid |
| `payment_consistency_90d` | `1 - std(days_overdue)/90` clipped |

### Target encoding (`src/features/target_encoding.py`)

| Feature | Notes |
|---------|-------|
| `purpose_target_enc` | fit on train period only, smoothing=20 |
| `home_ownership_target_enc` | same |

Maps are persisted to `artifacts/target_encoding.json` for online parity.  
**No noise** on inference transforms.

### Bureau (`src/features/bureau.py`)

Latest bureau report **before** application date:

| Feature | Formula |
|---------|---------|
| `bureau_balance_to_income` | `total_balance / income` |
| `inquiries_per_account` | `num_inquiries_6m / num_active_loans` |

## Leakage protection

- Aggregates & bureau use only pre-application events
- Target encoding fitted on train cutoff (`≤ 2022-12-31`)
- Time-based train/val/test split
- Leakage DAG checks: train/test ID overlap + future payments
