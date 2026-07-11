# tests/test_api.py

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_predictor():
    predictor = MagicMock()
    predictor.model_version = "test_v1"
    predictor.predict.return_value = MagicMock(
        application_id=1,
        pd_score=0.15,
        pd_calibrated=0.14,
        risk_bucket="MEDIUM",
        top_reasons=[],
        model_version="test_v1",
    )
    return predictor


@pytest.fixture
def client(mock_predictor):
    from src.serving.app import app
    import src.serving.app as app_module

    app_module.predictor = mock_predictor

    with TestClient(app) as c:
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
    assert "risk_bucket" in data
    assert data["risk_bucket"] in [
        "LOW", "MEDIUM", "HIGH", "VERY_HIGH"
    ]


def test_predict_invalid_credit_score(client):
    payload = {
        "application_id": 1,
        "loan_amount": 15000.0,
        "income": 60000.0,
        "loan_term": 36,
        "interest_rate": 12.5,
        "employment_years": 5.0,
        "credit_score": 1500,  # невалидный!
        "dti_ratio": 20.0,
        "num_open_accounts": 5,
        "num_delinquencies": 0,
        "total_credit_limit": 50000.0,
        "home_ownership": "RENT",
        "purpose": "debt_consolidation",
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation Error