# src/serving/app.py

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from loguru import logger
import mlflow
import pandas as pd

from src.serving.schemas import ScoringRequest, ScoringResponse
from src.serving.predict import Predictor
from src.config import MLflowConfig

predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    global predictor
    logger.info("Loading model...")
    mlflow.set_tracking_uri(MLflowConfig.TRACKING_URI)
    predictor = Predictor()
    predictor.load()
    logger.info("Model loaded successfully")
    yield
    # shutdown
    logger.info("Shutting down...")


app = FastAPI(
    title="Credit Scoring API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/predict", response_model=ScoringResponse)
async def predict(request: ScoringRequest):
    if predictor is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded"
        )

    try:
        result = predictor.predict(request)
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500, detail=str(e)
        )


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "model_version": predictor.model_version
        if predictor
        else None,
    }


@app.get("/model-info")
async def model_info():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return predictor.get_model_info()