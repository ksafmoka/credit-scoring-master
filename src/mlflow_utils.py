"""Helpers for robust MLflow usage from parallel Airflow tasks."""

from __future__ import annotations

import os
import time
from pathlib import Path

from src.config import MLflowConfig
from src.logging_utils import get_logger

logger = get_logger(__name__)

# Shared path: must be mounted RW in both mlflow + airflow containers
DEFAULT_ARTIFACT_ROOT = os.getenv(
    "MLFLOW_ARTIFACT_ROOT", "/mlflow/artifacts"
)


def ensure_artifact_dirs() -> Path:
    """Ensure local artifact root exists and is writable."""
    root = Path(DEFAULT_ARTIFACT_ROOT)
    try:
        root.mkdir(parents=True, exist_ok=True)
        test = root / ".write_test"
        test.write_text("ok")
        test.unlink(missing_ok=True)
    except Exception as exc:
        logger.warning(f"Artifact root {root} not writable: {exc}")
    return root


def ensure_experiment(name: str | None = None, tracking_uri: str | None = None) -> str:
    """
    Get-or-create experiment safely under concurrent task starts.

    Sets a shared file:// artifact location so Airflow workers can log models.
    """
    import mlflow
    from mlflow.exceptions import MlflowException, RestException
    from mlflow.tracking import MlflowClient

    ensure_artifact_dirs()

    uri = tracking_uri or MLflowConfig.TRACKING_URI
    exp_name = name or MLflowConfig.EXPERIMENT_SCORING
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    artifact_location = f"file://{DEFAULT_ARTIFACT_ROOT.rstrip('/')}/{exp_name}"

    def _location_ok(loc: str | None) -> bool:
        if not loc:
            return False
        bad = {"", "/mlflow", "file:/mlflow", "file:///mlflow"}
        return loc not in bad and not loc.rstrip("/").endswith("/mlflow")

    def _retire_broken(exp) -> None:
        """Rename+delete experiment with unusable artifact_location."""
        try:
            new_name = f"{exp_name}_broken_{int(time.time())}"
            client.rename_experiment(exp.experiment_id, new_name)
            client.delete_experiment(exp.experiment_id)
            logger.warning(
                f"Retired broken experiment id={exp.experiment_id} "
                f"artifact_location={exp.artifact_location!r}"
            )
        except Exception as exc:
            logger.warning(f"Could not retire broken experiment: {exc}")

    last_err: Exception | None = None
    for attempt in range(8):
        try:
            exp = client.get_experiment_by_name(exp_name)
            if exp is not None:
                if getattr(exp, "lifecycle_stage", "active") == "deleted":
                    try:
                        client.restore_experiment(exp.experiment_id)
                        exp = client.get_experiment(exp.experiment_id)
                    except Exception:
                        exp = None

            if exp is not None and not _location_ok(
                getattr(exp, "artifact_location", None)
            ):
                _retire_broken(exp)
                exp = None

            if exp is not None:
                mlflow.set_experiment(experiment_id=exp.experiment_id)
                return str(exp.experiment_id)

            experiment_id = client.create_experiment(
                exp_name, artifact_location=artifact_location
            )
            mlflow.set_experiment(experiment_id=experiment_id)
            logger.info(
                f"Created MLflow experiment '{exp_name}' id={experiment_id} "
                f"artifacts={artifact_location}"
            )
            return str(experiment_id)
        except (RestException, MlflowException, Exception) as exc:
            last_err = exc
            msg = str(exc).lower()
            if (
                "already exists" in msg
                or "resource_already_exists" in msg
                or "uniqueviolation" in msg
                or "duplicate key" in msg
            ):
                time.sleep(0.3 * (attempt + 1))
                exp = client.get_experiment_by_name(exp_name)
                if exp is not None and _location_ok(
                    getattr(exp, "artifact_location", None)
                ):
                    mlflow.set_experiment(experiment_id=exp.experiment_id)
                    return str(exp.experiment_id)
                if exp is not None:
                    _retire_broken(exp)
                continue
            if attempt < 7:
                time.sleep(0.5 * (attempt + 1))
                continue
            raise

    if last_err:
        raise last_err
    raise RuntimeError(f"Could not ensure MLflow experiment '{exp_name}'")


def log_model_safe(model, artifact_path: str = "model") -> bool:
    """
    Log sklearn-compatible model to MLflow.
    Returns True on success. Never raises — local artifacts are the source of truth for serving.
    """
    import mlflow
    import mlflow.sklearn

    try:
        ensure_artifact_dirs()
        mlflow.sklearn.log_model(model, artifact_path=artifact_path)
        return True
    except Exception as exc:
        logger.warning(
            f"mlflow.log_model failed ({exc}). "
            "Local artifacts/ still contain the model for API serving."
        )
        return False


def log_dir_safe(local_dir: str | Path, artifact_path: str = "scoring_bundle") -> bool:
    import mlflow

    try:
        ensure_artifact_dirs()
        mlflow.log_artifacts(str(local_dir), artifact_path=artifact_path)
        return True
    except Exception as exc:
        logger.warning(f"mlflow.log_artifacts failed ({exc})")
        return False
