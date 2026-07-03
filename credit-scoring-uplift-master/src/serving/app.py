# src/serving/app.py

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from loguru import logger
import mlflow

from src.serving.schemas import ScoringRequest, ScoringResponse
from src.serving.predict import Predictor
from src.config import MLflowConfig

predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    logger.info("Starting API...")
    mlflow.set_tracking_uri(MLflowConfig.TRACKING_URI)
    predictor = Predictor()
    predictor.load()  # НЕ падает если модели нет

    if predictor.is_ready():
        logger.info(
            f"API started. Model version: {predictor.model_version}"
        )
    else:
        logger.warning(
            "API started WITHOUT a model. "
            "Run training pipeline first, then POST /reload"
        )

    yield

    logger.info("Shutting down API...")


app = FastAPI(
    title="Credit Scoring API",
    version="1.0.0",
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
                "Model not loaded. "
                "Run training pipeline first, then POST /reload"
            ),
        )
    try:
        return predictor.predict(request)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload")
async def reload_model():
    """
    Перезагрузить модель из MLflow без рестарта контейнера.
    Вызывать после успешного завершения dag_training.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")

    predictor.reload()

    if predictor.is_ready():
        return {
            "status": "ok",
            "model_version": predictor.model_version,
        }
    else:
        raise HTTPException(
            status_code=503,
            detail="Model still not available after reload",
        )


@app.get("/model-info")
async def model_info():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    return predictor.get_model_info()