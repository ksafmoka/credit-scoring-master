\c credit_scoring;

SET ROLE ml_user;

-- Use DOUBLE PRECISION for engineered features to avoid NUMERIC overflow
-- (e.g. income * credit_score can exceed NUMERIC(14,6) max of 1e8).
CREATE TABLE IF NOT EXISTS features.application_features (
    application_id              BIGINT PRIMARY KEY,
    feature_date                DATE NOT NULL,

    loan_to_income              DOUBLE PRECISION,
    credit_utilization          DOUBLE PRECISION,
    income_log                  DOUBLE PRECISION,
    loan_amount_log             DOUBLE PRECISION,
    dti_ratio_clipped           DOUBLE PRECISION,
    employment_years            DOUBLE PRECISION,
    credit_score_norm           DOUBLE PRECISION,
    num_open_accounts           DOUBLE PRECISION,
    num_delinquencies           DOUBLE PRECISION,
    interest_rate               DOUBLE PRECISION,
    loan_amount_x_dti           DOUBLE PRECISION,
    income_x_credit_score       DOUBLE PRECISION,

    dti_bucket                  VARCHAR(20),
    credit_score_bucket         VARCHAR(20),

    avg_days_overdue_30d        DOUBLE PRECISION,
    avg_days_overdue_90d        DOUBLE PRECISION,
    avg_days_overdue_180d       DOUBLE PRECISION,
    max_days_overdue_90d        DOUBLE PRECISION,
    pct_late_payments_90d       DOUBLE PRECISION,
    total_paid_90d              DOUBLE PRECISION,
    payment_consistency_90d     DOUBLE PRECISION,

    bureau_balance_to_income    DOUBLE PRECISION,
    inquiries_per_account       DOUBLE PRECISION,
    avg_account_age_months      DOUBLE PRECISION,

    purpose_target_enc          DOUBLE PRECISION,
    home_ownership_target_enc   DOUBLE PRECISION,

    feature_version             VARCHAR(50) NOT NULL,
    computed_at                 TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feat_date
    ON features.application_features (feature_date);
CREATE INDEX IF NOT EXISTS idx_feat_version
    ON features.application_features (feature_version);

RESET ROLE;
