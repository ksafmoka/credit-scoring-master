\c credit_scoring;

CREATE TABLE IF NOT EXISTS predictions.scoring_predictions (
    prediction_id   BIGSERIAL PRIMARY KEY,
    application_id  BIGINT NOT NULL,
    model_version   VARCHAR(100) NOT NULL,
    pd_score        NUMERIC(10, 8),
    pd_calibrated   NUMERIC(10, 8),
    risk_bucket     VARCHAR(20),
    shap_top3       JSONB,
    predicted_at    TIMESTAMP DEFAULT NOW()
);

-- CREATE TABLE IF NOT EXISTS predictions.uplift_predictions (
--     prediction_id       BIGSERIAL PRIMARY KEY,
--     client_id           BIGINT NOT NULL,
--     model_version       VARCHAR(100) NOT NULL,
--     uplift_score        NUMERIC(10, 8),
--     segment             VARCHAR(30),
--     recommended_action  VARCHAR(50),
--     predicted_at        TIMESTAMP DEFAULT NOW()
-- );

-- ─────────────────────────────────────────
-- Monitoring tables
-- ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS monitoring.feature_drift (
    check_id            BIGSERIAL PRIMARY KEY,
    feature_name        VARCHAR(100) NOT NULL,
    check_date          DATE NOT NULL,
    psi_value           NUMERIC(10, 6),
    ks_statistic        NUMERIC(10, 6),
    ks_pvalue           NUMERIC(10, 6),
    is_drifted          BOOLEAN,
    reference_period    VARCHAR(50),
    current_period      VARCHAR(50),
    checked_at          TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS monitoring.model_performance (
    check_id            BIGSERIAL PRIMARY KEY,
    model_version       VARCHAR(100) NOT NULL,
    check_date          DATE NOT NULL,
    auc_roc             NUMERIC(8, 6),
    auc_pr              NUMERIC(8, 6),
    brier_score         NUMERIC(8, 6),
    ks_statistic        NUMERIC(8, 6),
    num_predictions     INT,
    checked_at          TIMESTAMP DEFAULT NOW()
);