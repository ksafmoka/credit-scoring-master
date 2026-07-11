\c credit_scoring;

CREATE TABLE IF NOT EXISTS features.application_features (
    application_id              BIGINT PRIMARY KEY,
    feature_date                DATE NOT NULL,

    -- числовые трансформации
    loan_to_income              NUMERIC(10, 6),
    credit_utilization          NUMERIC(10, 6),
    income_log                  NUMERIC(10, 6),
    loan_amount_log             NUMERIC(10, 6),
    dti_ratio_clipped           NUMERIC(6, 2),

    -- бакеты
    dti_bucket                  VARCHAR(20),
    credit_score_bucket         VARCHAR(20),

    -- агрегаты из payment_history (окна 30/90/180 дней)
    avg_days_overdue_30d        NUMERIC(8, 4),
    avg_days_overdue_90d        NUMERIC(8, 4),
    avg_days_overdue_180d       NUMERIC(8, 4),
    max_days_overdue_90d        NUMERIC(8, 4),
    pct_late_payments_90d       NUMERIC(6, 4),
    total_paid_90d              NUMERIC(14, 2),
    payment_consistency_90d     NUMERIC(6, 4),

    -- агрегаты из credit_bureau
    bureau_balance_to_income    NUMERIC(10, 6),
    inquiries_per_account       NUMERIC(8, 4),
    avg_account_age_months      NUMERIC(8, 2),

    -- target encoding
    purpose_target_enc          NUMERIC(10, 8),
    home_ownership_target_enc   NUMERIC(10, 8),

    -- кросс-фичи
    loan_amount_x_dti           NUMERIC(14, 6),
    income_x_credit_score       NUMERIC(14, 6),

    -- метаданные
    feature_version             VARCHAR(50) NOT NULL,
    computed_at                 TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feat_date
    ON features.application_features (feature_date);
CREATE INDEX IF NOT EXISTS idx_feat_version
    ON features.application_features (feature_version);