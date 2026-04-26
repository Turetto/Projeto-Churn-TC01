import pytest
from fastapi.testclient import TestClient
from src.api import app

VALID_PAYLOAD = {
  "Contract": "Month-to-month",
  "Dependents": "No",
  "DeviceProtection": "No",
  "InternetService": "Fiber optic",
  "MonthlyCharges": 65.5,
  "MultipleLines": "No",
  "OnlineBackup": "No",
  "OnlineSecurity": "No",
  "PaperlessBilling": "Yes",
  "Partner": "Yes",
  "PaymentMethod": "Electronic check",
  "PhoneService": "Yes",
  "StreamingMovies": "No",
  "StreamingTV": "No",
  "TechSupport": "No",
  "TotalCharges": 786,
  "gender": "Female",
  "tenure": 12
}

@pytest.fixture(scope="module")
def client():
    """Cria o TestClient disparando os eventos de startup e shutdown."""
    with TestClient(app) as c:
        yield c

def test_health_endpoint(client):
    """Endpoint /health deve retornar status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint_status(client):
    """Endpoint /predict deve retornar status 200."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200


def test_predict_response_schema(client):
    """Resposta do /predict deve ter todos os campos esperados."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    data = response.json()

    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert "risk_level" in data
    assert "latency_ms" in data


def test_predict_probability_range(client):
    """Probabilidade de churn deve estar entre 0 e 1."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    data = response.json()

    assert 0.0 <= data["churn_probability"] <= 1.0


def test_predict_risk_level_valid(client):
    """Risk level deve ser low, medium ou high."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    data = response.json()

    assert data["risk_level"] in ["low", "medium", "high"]


def test_predict_invalid_payload(client):
    """Payload inválido deve retornar status 422."""
    response = client.post("/predict", json={"tenure": -1})
    assert response.status_code == 422