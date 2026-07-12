\c credit_scoring;

SET ROLE ml_user;

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

CREATE INDEX IF NOT EXISTS idx_pred_app
    ON predictions.scoring_predictions (application_id);
CREATE INDEX IF NOT EXISTS idx_pred_time
    ON predictions.scoring_predictions (predicted_at);

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

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA predictions TO ml_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA monitoring TO ml_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA predictions TO ml_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA monitoring TO ml_user;
