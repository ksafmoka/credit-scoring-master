"""FastAPI application for PD scoring."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from src.logging_utils import get_logger
logger = get_logger(__name__)

from src.config import MLflowConfig
from src.serving.predict import Predictor
from src.serving.schemas import ScoringRequest, ScoringResponse

predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    logger.info("Starting Credit Scoring API...")
    try:
        import mlflow

        mlflow.set_tracking_uri(MLflowConfig.TRACKING_URI)
    except Exception as exc:  # pragma: no cover
        logger.warning(f"MLflow unavailable at startup: {exc}")
    predictor = Predictor()
    predictor.load()

    if predictor.is_ready():
        logger.info(f"API started. Model version: {predictor.model_version}")
    else:
        logger.warning(
            "API started WITHOUT a model. "
            "Run training pipeline first, then POST /reload"
        )
    yield
    logger.info("Shutting down API...")


app = FastAPI(
    title="Credit Scoring API",
    description="Probability of default (PD) scoring service",
    version="1.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": predictor.is_ready() if predictor else False,
        "model_version": predictor.model_version if predictor else None,
    }


@app.post("/predict", response_model=ScoringResponse)
async def predict(request: ScoringRequest):
    if predictor is None or not predictor.is_ready():
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. Run training pipeline first, then POST /reload"
            ),
        )
    try:
        return predictor.predict(request)
    except Exception as exc:
        logger.error(f"Prediction error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/reload")
async def reload_model():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    predictor.reload()
    if predictor.is_ready():
        return {"status": "ok", "model_version": predictor.model_version}
    raise HTTPException(
        status_code=503,
        detail="Model still not available after reload",
    )


@app.get("/model-info")
async def model_info():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    return predictor.get_model_info()
