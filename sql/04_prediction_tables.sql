\c credit_scoring;

SET ROLE ml_user;

CREATE TABLE IF NOT EXISTS predictions.scoring_predictions (
    prediction_id   BIGSERIAL PRIMARY KEY,
    application_id  BIGINT NOT NULL,
    model_version   VARCHAR(100) NOT NULL,
    pd_score        DOUBLE PRECISION,
    pd_calibrated   DOUBLE PRECISION,
    risk_bucket     VARCHAR(20),
    shap_top3       JSONB,
    predicted_at    TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pred_app
    ON predictions.scoring_predictions (application_id);
CREATE INDEX IF NOT EXISTS idx_pred_time
    ON predictions.scoring_predictions (predicted_at);

CREATE TABLE IF NOT EXISTS monitoring.feature_drift (
    check_id            BIGSERIAL PRIMARY KEY,
    feature_name        VARCHAR(100) NOT NULL,
    check_date          DATE NOT NULL,
    psi_value           DOUBLE PRECISION,
    ks_statistic        DOUBLE PRECISION,
    ks_pvalue           DOUBLE PRECISION,
    is_drifted          BOOLEAN,
    reference_period    VARCHAR(50),
    current_period      VARCHAR(50),
    checked_at          TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS monitoring.model_performance (
    check_id            BIGSERIAL PRIMARY KEY,
    model_version       VARCHAR(100) NOT NULL,
    check_date          DATE NOT NULL,
    auc_roc             DOUBLE PRECISION,
    auc_pr              DOUBLE PRECISION,
    brier_score         DOUBLE PRECISION,
    ks_statistic        DOUBLE PRECISION,
    num_predictions     INT,
    checked_at          TIMESTAMP DEFAULT NOW()
);

-- Feature-drift Telegram queue (host notifier if Docker blocks api.telegram.org)
CREATE TABLE IF NOT EXISTS monitoring.alert_queue (
    alert_id   BIGSERIAL PRIMARY KEY,
    message    TEXT NOT NULL,
    status     VARCHAR(20) NOT NULL DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT NOW(),
    sent_at    TIMESTAMP,
    error      TEXT
);

RESET ROLE;
