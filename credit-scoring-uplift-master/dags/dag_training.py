# dags/dag_training.py

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.task_group import TaskGroup

default_args = {
    "owner": "ml-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

dag = DAG(
    dag_id="model_training",
    default_args=default_args,
    description="Train scoring + uplift models",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["training", "ml"],
)


def prepare_datasets(**context):
    from src.models.scoring.train import prepare_time_split
    from src.config import get_db_engine
    import json

    engine = get_db_engine()
    info = prepare_time_split(engine)
    context["ti"].xcom_push(
        key="split_info", value=json.dumps(info)
    )


def train_single_model(model_type, **context):
    from src.models.scoring.train import train_model
    from src.models.scoring.hyperopt import optimize_hyperparams
    from src.config import get_db_engine
    import mlflow

    engine = get_db_engine()
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("credit_scoring")

    with mlflow.start_run(run_name=f"{model_type}_{context['ds']}"):
        best_params = optimize_hyperparams(
            model_type=model_type,
            engine=engine,
            n_trials=30,
        )
        model, metrics = train_model(
            model_type=model_type,
            params=best_params,
            engine=engine,
        )
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)

        context["ti"].xcom_push(
            key=f"{model_type}_auc",
            value=metrics["val_auc_roc"],
        )


def train_ensemble(**context):
    from src.models.scoring.ensemble import StackingEnsemble
    from src.config import get_db_engine
    import mlflow

    engine = get_db_engine()
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("credit_scoring")

    with mlflow.start_run(run_name=f"ensemble_{context['ds']}"):
        ensemble = StackingEnsemble(
            base_models=["catboost", "lightgbm", "xgboost"],
        )
        model, metrics = ensemble.fit(engine)
        mlflow.log_metrics(metrics)

        context["ti"].xcom_push(
            key="ensemble_auc", value=metrics["val_auc_roc"]
        )


def run_leakage_check(**context):
    from src.models.scoring.evaluate import leakage_check
    from src.config import get_db_engine

    engine = get_db_engine()
    result = leakage_check(
        engine, checks=["train_test_overlap"]
    )
    if not result.passed:
        raise ValueError(f"LEAKAGE: {result.details}")


def decide_register(**context):
    auc = context["ti"].xcom_pull(
        task_ids="ensemble", key="ensemble_auc"
    )
    return "register" if auc and auc > 0.75 else "skip"


def register_model(**context):
    import mlflow

    mlflow.set_tracking_uri("http://mlflow:5000")
    # регистрация лучшей модели
    pass


with dag:
    t_prepare = PythonOperator(
        task_id="prepare_datasets",
        python_callable=prepare_datasets,
    )

    with TaskGroup("base_models") as base_group:
        for m in ["catboost", "lightgbm", "xgboost"]:
            PythonOperator(
                task_id=f"train_{m}",
                python_callable=train_single_model,
                op_kwargs={"model_type": m},
            )

    t_ensemble = PythonOperator(
        task_id="ensemble",
        python_callable=train_ensemble,
    )

    t_leakage = PythonOperator(
        task_id="leakage_check",
        python_callable=run_leakage_check,
    )

    t_decide = BranchPythonOperator(
        task_id="decide",
        python_callable=decide_register,
    )

    t_register = PythonOperator(
        task_id="register",
        python_callable=register_model,
    )

    t_skip = PythonOperator(
        task_id="skip",
        python_callable=lambda: print("Not good enough"),
    )

    (
        t_prepare
        >> base_group
        >> t_ensemble
        >> t_leakage
        >> t_decide
        >> [t_register, t_skip]
    )