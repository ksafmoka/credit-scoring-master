\c credit_scoring;

-- ─────────────────────────────────────────
-- Заявки на кредит
-- ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS raw.applications (
    application_id      BIGSERIAL PRIMARY KEY,
    client_id           BIGINT NOT NULL,
    application_date    DATE NOT NULL,
    loan_amount         NUMERIC(12, 2),
    loan_term           INT,
    interest_rate       NUMERIC(5, 2),
    income              NUMERIC(12, 2),
    employment_years    NUMERIC(4, 1),
    home_ownership      VARCHAR(20),
    purpose             VARCHAR(50),
    dti_ratio           NUMERIC(5, 2),
    credit_score        INT,
    num_open_accounts   INT,
    num_delinquencies   INT,
    total_credit_limit  NUMERIC(14, 2),
    -- target
    is_default          BOOLEAN,
    -- для uplift: получил ли клиент предложение рефинансирования
    treatment_flag      BOOLEAN DEFAULT FALSE,
    -- метаданные загрузки
    loaded_at           TIMESTAMP DEFAULT NOW(),
    data_source         VARCHAR(50) DEFAULT 'lending_club'
);

CREATE INDEX IF NOT EXISTS idx_app_client
    ON raw.applications (client_id);
CREATE INDEX IF NOT EXISTS idx_app_date
    ON raw.applications (application_date);
CREATE INDEX IF NOT EXISTS idx_app_default
    ON raw.applications (is_default);

-- ─────────────────────────────────────────
-- История платежей
-- ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS raw.payment_history (
    payment_id          BIGSERIAL PRIMARY KEY,
    application_id      BIGINT REFERENCES raw.applications (application_id),
    payment_date        DATE NOT NULL,
    amount_due          NUMERIC(10, 2),
    amount_paid         NUMERIC(10, 2),
    days_overdue        INT DEFAULT 0,
    loaded_at           TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_payment_app
    ON raw.payment_history (application_id);
CREATE INDEX IF NOT EXISTS idx_payment_date
    ON raw.payment_history (payment_date);

-- ─────────────────────────────────────────
-- Кредитное бюро
-- ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS raw.credit_bureau (
    bureau_id               BIGSERIAL PRIMARY KEY,
    client_id               BIGINT NOT NULL,
    report_date             DATE NOT NULL,
    num_inquiries_6m        INT,
    num_active_loans        INT,
    total_balance           NUMERIC(14, 2),
    num_defaults_hist       INT,
    oldest_account_months   INT,
    loaded_at               TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_bureau_client
    ON raw.credit_bureau (client_id);
CREATE INDEX IF NOT EXISTS idx_bureau_date
    ON raw.credit_bureau (report_date);