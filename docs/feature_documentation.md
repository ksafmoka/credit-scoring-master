# Feature documentation / Документация признаков

## Feature groups / Группы признаков

### Numerical & cross (`src/features/numerical.py`) — used by both models / Числовые и кросс-признаки — обе модели

| Feature / Признак | Description / Описание | Formula / Формула |
|---------|------------------|----------|
| `loan_to_income` | Отношение займа к доходу / Loan-to-income ratio | `loan_amount / income` |
| `credit_utilization` | Кредитная утилизация / Credit utilization ratio | `loan_amount / total_credit_limit` |
| `income_log` | Логарифм дохода / Log-transformed income | `log1p(income)` |
| `loan_amount_log` | Логарифм суммы займа / Log-transformed loan amount | `log1p(loan_amount)` |
| `dti_ratio_clipped` | clipped / Обрезанный долг-доход | `clip(dti_ratio, 0, 100)` |
| `employment_years` | Стаж работы / Employment tenure | employment tenure |
| `credit_score_norm` | Нормализованный кредитный рейтинг / Normalized credit score | `(credit_score - 300) / 550` clipped to [0, 1] |
| `num_open_accounts` | Кол-во открытых счетов / Number of open accounts | open trade lines |
| `num_delinquencies` | Кол-во просрочек / Number of delinquencies | delinquency count |
| `interest_rate` | Процентная ставка / Loan interest rate (APR) | loan APR |
| `loan_term` | Срок займа / Loan term in months | 36 or 60 months |
| `num_inquiries_6m` | Запросы за 6 мес / Credit inquiries in last 6 months | credit inquiries |
| `loan_amount_x_dti` | Сумма × DTI / Loan amount × debt-to-income | `loan_amount × dti` |
| `income_x_credit_score` | Доход × Рейтинг / Income × credit score | `income × credit_score` |
| `dti_x_credit_score` | DTI × Рейтинг / Debt-to-income × credit score | `dti × credit_score_norm` |
| `loan_amount_x_interest_rate` | Сумма × Ставка / Loan amount × interest rate | `loan_amount × interest_rate` |

### Aggregation (`src/features/aggregations.py`) — Model A (with_history) only / Агрегаты — только модель А

Only payments with `payment_date < application_date` / Только платежи до подачи заявки.

| Feature / Признак | Description / Описание | Window / Окно |
|---------|--------|--------|
| `avg_days_overdue_30d` | Ср. просрочка за 30 дней / Mean overdue days (30d) | 30 дней / 30 days |
| `avg_days_overdue_90d` | Ср. просрочка за 90 дней / Mean overdue days (90d) | 90 дней / 90 days |
| `avg_days_overdue_180d` | Ср. просрочка за 180 дней / Mean overdue days (180d) | 180 дней / 180 days |
| `max_days_overdue_90d` | Макс. просрочка за 90 дней / Max overdue (90d) | 90 дней / 90 days |
| `pct_late_payments_90d` | Доля просроченных платежей / Share of late payments | 90 дней / 90 days |
| `total_paid_90d` | Всего выплачено / Total amount paid | 90 дней / 90 days |
| `payment_consistency_90d` | Стабильность платежей / Payment consistency score | `1 - std(days_overdue)/90` clipped |

### Target encoding (`src/features/target_encoding.py`) — used by both models / Target encoding — обе модели

| Feature / Признак | Description / Описание | Notes / Заметки |
|---------|-------|-------|
| `purpose_target_enc` | Цель займа (закодированная) / Loan purpose (encoded) | fit on train only, smoothing=20 |
| `home_ownership_target_enc` | Тип жилья (закодированный) / Home ownership (encoded) | same / аналогично |

Maps are persisted to `artifacts/target_encoding.json` for online parity.
**No noise** on inference transforms / **Без шума** при инференсе.

### Bureau (`src/features/bureau.py`) — used by both models / Бюро кредитных историй — обе модели

Latest bureau report **before** application date / Последняя запись бюро **до** даты заявки:

| Feature / Признак | Description / Описание | Formula / Формула |
|---------|---------|---------|
| `bureau_balance_to_income` | Баланс бюро к доходу / Bureau balance to income ratio | `total_balance / income` |
| `inquiries_per_account` | Запросы на счёт / Inquiries per active account | `num_inquiries_6m / num_active_loans` |

## Feature sets per model / Наборы признаков по моделям

| Feature set / Набор | Config | Model A / Модель А | Model B / Модель Б |
|-------------|--------|---------|---------|
| Numerical (16) / Числовые | `FeatureConfig.NUMERICAL_FEATURES` | ✅ | ✅ |
| Target encoding (2) / TE | `FeatureConfig.TARGET_ENCODED_FEATURES` | ✅ | ✅ |
| Aggregation (7) / Агрегаты | `FeatureConfig.AGGREGATION_FEATURES` | ✅ | ❌ |
| Bureau (2) / Бюро | `FeatureConfig.BUREAU_FEATURES` | ✅ | ✅ |
| **Total / Итого** | `ALL_FEATURES` / `COLD_START_FEATURES` | **27** | **20** |

## Leakage protection / Защита от утечек

- Aggregates & bureau use only pre-application events / Агрегаты и бюро используют только данные до заявки
- Target encoding fitted on train cutoff only / Target encoding обучается только на train периоде
- Time-based train/val/test split (justified by EDA statistical tests) / Временной сплит (подтверждён стат. тестами EDA)
- Leakage DAG checks: train/test ID overlap + future payments / Проверки: пересечение ID + будущие платежи
- fe_ids consistency ensures all FE tasks operate on the same application set / fe_ids гарантирует единый набор ID
