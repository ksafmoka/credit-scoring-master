# Feature Documentation

## Feature Groups

### Numerical (src/features/numerical.py)
| Feature | Formula | Source |
|---------|---------|--------|
| loan_to_income | loan_amount / income | raw.applications |
| credit_utilization | loan_amount / total_credit_limit | raw.applications |
| income_log | log1p(income) | raw.applications |
| loan_amount_log | log1p(loan_amount) | raw.applications |
| dti_ratio_clipped | clip(dti_ratio, 0, 100) | raw.applications |

### Aggregation (src/features/aggregations.py)
| Feature | Window | Source |
|---------|--------|--------|
| avg_days_overdue_{W}d | 30/90/180 | raw.payment_history |
| max_days_overdue_90d | 90 | raw.payment_history |
| pct_late_payments_90d | 90 | raw.payment_history |
| total_paid_90d | 90 | raw.payment_history |

### Target Encoded (src/features/target_encoding.py)
| Feature | Smoothing |
|---------|-----------|
| purpose_target_enc | 20 |
| home_ownership_target_enc | 20 |

### Cross Features
| Feature | Formula |
|---------|---------|
| loan_amount_x_dti | loan_amount × dti_ratio |
| income_x_credit_score | income × credit_score |

## Leakage Protection
- All aggregation features use ONLY payments BEFORE application_date
- Target encoding fitted ONLY on train period (≤ 2022-12-31)
- Post-origination columns excluded (see tests/test_leakage.py)