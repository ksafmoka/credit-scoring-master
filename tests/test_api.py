"""API tests with mocked predictor (no MLflow required)."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.serving.schemas import ScoringResponse


@pytest.fixture
def mock_predictor():
    predictor = MagicMock()
    predictor.model_version = "test_v1"
    predictor.is_ready.return_value = True
    predictor.predict.return_value = ScoringResponse(
        application_id=1,
        pd_score=0.15,
        pd_calibrated=0.14,
        risk_bucket="MEDIUM",
        top_reasons=[],
        model_version="test_v1",
    )
    predictor.get_model_info.return_value = {
        "model_version": "test_v1",
        "model_loaded": True,
        "feature_count": 3,
        "features": ["a", "b", "c"],
    }
    return predictor


@pytest.fixture
def client(mock_predictor):
    # Avoid lifespan model loading
    with patch("src.serving.app.Predictor") as PredCls:
        PredCls.return_value = mock_predictor
        from src.serving.app import app
        import src.serving.app as app_module

        app_module.predictor = mock_predictor
        with TestClient(app, raise_server_exceptions=False) as c:
            # lifespan may overwrite; force mock again
            app_module.predictor = mock_predictor
            yield c


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_predict_valid_request(client):
    payload = {
        "application_id": 1,
        "loan_amount": 15000.0,
        "income": 60000.0,
        "loan_term": 36,
        "interest_rate": 12.5,
        "employment_years": 5.0,
        "credit_score": 700,
        "dti_ratio": 20.0,
        "num_open_accounts": 5,
        "num_delinquencies": 0,
        "total_credit_limit": 50000.0,
        "home_ownership": "RENT",
        "purpose": "debt_consolidation",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "pd_score" in data
    assert data["risk_bucket"] in ["LOW", "MEDIUM", "HIGH", "VERY_HIGH"]


def test_predict_invalid_credit_score(client):
    payload = {
        "application_id": 1,
        "loan_amount": 15000.0,
        "income": 60000.0,
        "loan_term": 36,
        "interest_rate": 12.5,
        "employment_years": 5.0,
        "credit_score": 1500,
        "dti_ratio": 20.0,
        "num_open_accounts": 5,
        "num_delinquencies": 0,
        "total_credit_limit": 50000.0,
        "home_ownership": "RENT",
        "purpose": "debt_consolidation",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_without_model_returns_503():
    dead = MagicMock()
    dead.is_ready.return_value = False
    with patch("src.serving.app.Predictor") as PredCls:
        PredCls.return_value = dead
        from src.serving.app import app
        import src.serving.app as app_module

        app_module.predictor = dead
        with TestClient(app, raise_server_exceptions=False) as c:
            app_module.predictor = dead
            payload = {
                "application_id": 1,
                "loan_amount": 15000.0,
                "income": 60000.0,
                "loan_term": 36,
                "interest_rate": 12.5,
                "employment_years": 5.0,
                "credit_score": 700,
                "dti_ratio": 20.0,
                "num_open_accounts": 5,
                "num_delinquencies": 0,
                "total_credit_limit": 50000.0,
                "home_ownership": "RENT",
                "purpose": "debt_consolidation",
            }
            response = c.post("/predict", json=payload)
            assert response.status_code == 503
