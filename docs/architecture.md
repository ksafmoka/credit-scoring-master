# Architecture

## System Components

```text
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│ Lending Club │────→│   PostgreSQL     │────→│   Grafana    │
│   CSV        │     │  raw / features  │     │  dashboards  │
└─────────────┘     │  predictions     │     └─────────────┘
                    │  monitoring      │
                    └───────┬──────────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
        ┌─────┴─────┐ ┌────┴────┐ ┌──────┴──────┐
        │  Airflow   │ │ MLflow  │ │  FastAPI    │
        │  DAGs      │ │ Track   │ │  Serving    │
        └────────────┘ └─────────┘ └─────────────┘
    Data Flow
1. dag_data_ingestion: CSV → raw.applications + raw.payment_history
2. dag_feature_engineering: raw → features.application_features
3. dag_training: features → MLflow model registry
4. dag_batch_prediction: features + model → predictions.scoring_predictions
5. dag_monitoring: features + predictions → monitoring.feature_drift + model_performance
    Databases
** credit_scoring: application data, features, predictions, monitoring
** airflow: Airflow metadata
** mlflow: MLflow backend store