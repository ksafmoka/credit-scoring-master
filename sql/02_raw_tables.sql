\c credit_scoring;

-- Create as ml_user so tables + sequences are owned by the app role.
-- DOUBLE PRECISION everywhere money/ratio columns can be huge in Lending Club.
SET ROLE ml_user;

CREATE TABLE IF NOT EXISTS raw.applications (
    application_id      BIGSERIAL PRIMARY KEY,
    client_id           BIGINT NOT NULL,
    application_date    DATE NOT NULL,
    loan_amount         DOUBLE PRECISION,
    loan_term           INT,
    interest_rate       DOUBLE PRECISION,
    income              DOUBLE PRECISION,
    employment_years    DOUBLE PRECISION,
    home_ownership      VARCHAR(20),
    purpose             VARCHAR(50),
    dti_ratio           DOUBLE PRECISION,
    credit_score        DOUBLE PRECISION,
    num_open_accounts   INT,
    num_delinquencies   INT,
    total_credit_limit  DOUBLE PRECISION,
    is_default          BOOLEAN,
    loaded_at           TIMESTAMP DEFAULT NOW(),
    data_source         VARCHAR(50) DEFAULT 'lending_club'
);

CREATE INDEX IF NOT EXISTS idx_app_client ON raw.applications (client_id);
CREATE INDEX IF NOT EXISTS idx_app_date ON raw.applications (application_date);
CREATE INDEX IF NOT EXISTS idx_app_default ON raw.applications (is_default);

CREATE TABLE IF NOT EXISTS raw.payment_history (
    payment_id          BIGSERIAL PRIMARY KEY,
    application_id      BIGINT REFERENCES raw.applications (application_id),
    payment_date        DATE NOT NULL,
    amount_due          DOUBLE PRECISION,
    amount_paid         DOUBLE PRECISION,
    days_overdue        INT DEFAULT 0,
    loaded_at           TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_payment_app ON raw.payment_history (application_id);
CREATE INDEX IF NOT EXISTS idx_payment_date ON raw.payment_history (payment_date);

CREATE TABLE IF NOT EXISTS raw.credit_bureau (
    bureau_id               BIGSERIAL PRIMARY KEY,
    client_id               BIGINT NOT NULL,
    report_date             DATE NOT NULL,
    num_inquiries_6m        INT,
    num_active_loans        INT,
    total_balance           DOUBLE PRECISION,
    num_defaults_hist       INT,
    oldest_account_months   INT,
    loaded_at               TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_bureau_client ON raw.credit_bureau (client_id);
CREATE INDEX IF NOT EXISTS idx_bureau_date ON raw.credit_bureau (report_date);

RESET ROLE;
