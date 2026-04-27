import logging
import time
from pathlib import Path

import numpy as np
import torch
import mlflow.pytorch
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from src.features import build_preprocessor, load_raw_data, split_features_target
from src.train import get_device, set_seeds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Configurações
SEED = 1312
DB_PATH = Path("models/mlflow.db").resolve()
MLFLOW_TRACKING_URI = f"sqlite:///{DB_PATH}"
MODEL_NAME = "ChurnMLP_v1"
THRESHOLD = 0.30

# Inicializar app
app = FastAPI(
    title="Churn Forecast API",
    description="API de inferência para predição de churn — Projeto-Churn-TC01",
    version="0.1.0",
)


# Schema de entrada
class CustomerFeatures(BaseModel):
    tenure: float = Field(..., ge=0, description="Meses de contrato")
    MonthlyCharges: float = Field(..., ge=0, description="Cobrança mensal")
    TotalCharges: float = Field(..., ge=0, description="Cobrança total")
    gender: str = Field(..., description="Female ou Male")
    Partner: str = Field(..., description="Yes ou No")
    Dependents: str = Field(..., description="Yes ou No")
    PhoneService: str = Field(..., description="Yes ou No")
    MultipleLines: str = Field(..., description="Yes, No ou No phone service")
    InternetService: str = Field(..., description="DSL, Fiber optic ou No")
    OnlineSecurity: str = Field(..., description="Yes, No ou No internet service")
    OnlineBackup: str = Field(..., description="Yes, No ou No internet service")
    DeviceProtection: str = Field(..., description="Yes, No ou No internet service")
    TechSupport: str = Field(..., description="Yes, No ou No internet service")
    StreamingTV: str = Field(..., description="Yes, No ou No internet service")
    StreamingMovies: str = Field(..., description="Yes, No ou No internet service")
    Contract: str = Field(..., description="Month-to-month, One year ou Two year")
    PaperlessBilling: str = Field(..., description="Yes ou No")
    PaymentMethod: str = Field(
        ...,
        description="Electronic check, Mailed check, Bank transfer ou Credit card"
    )

    model_config = {"json_schema_extra": {"example": {
        "tenure": 12,
        "MonthlyCharges": 65.5,
        "TotalCharges": 786.0,
        "gender": "Female",
        "Partner": "Yes",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
    }}}


# Schema de saída
class PredictionResponse(BaseModel):
    churn_probability: float = Field(..., description="Probabilidade de churn [0, 1]")
    churn_prediction: int = Field(..., description="Predição binária (0=não, 1=sim)")
    risk_level: str = Field(..., description="low, medium ou high")
    latency_ms: float = Field(..., description="Latência da inferência em ms")


# Carregar modelo e preprocessor no startup
@app.on_event("startup")
async def load_model() -> None:
    global model, preprocessor, device

    logger.info("Iniciando carregamento do modelo...")
    set_seeds(SEED)
    device = get_device()

    # Carregar preprocessor — fit nos dados de treino
    try:
        df = load_raw_data(Path("data/raw/telco_churn.csv"))
        X, _ = split_features_target(df)
        preprocessor = build_preprocessor()
        preprocessor.fit(X)
        logger.info("Preprocessor carregado e fitado.")
    except Exception as e:
        logger.error("Erro ao carregar preprocessor: %s", e)
        raise

    # Carregar modelo do MLflow
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        runs = mlflow.search_runs(
            experiment_names=["churn-mlp"],
            order_by=["metrics.roc_auc DESC"],
        )
        if runs.empty:
            raise ValueError("Nenhum run encontrado no experimento churn-mlp.")

        best_run_id = runs.iloc[0]["run_id"]
        model_uri = f"runs:/{best_run_id}/model"
        model = mlflow.pytorch.load_model(model_uri, map_location=device)
        model.eval()
        logger.info("Modelo carregado do run: %s", best_run_id)
        logger.info("Threshold de decisão: %.2f", THRESHOLD)
    except Exception as e:
        logger.error("Erro ao carregar modelo: %s", e)
        raise


# Middleware de latência
# teste para logging de latência
@app.middleware("http")
async def log_latency(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    latency = (time.time() - start) * 1000
    logger.info("%-6s %-20s → %d (%.1fms)",
                request.method, request.url.path,
                response.status_code, latency)
    return response


# Endpoints
@app.get("/health", tags=["monitoring"])
async def health() -> dict:
    """Verifica se a API está operacional."""
    return {"status": "ok", "model": MODEL_NAME, "version": "0.1.0"}


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(customer: CustomerFeatures) -> PredictionResponse:
    """
    Recebe features de um cliente e retorna a probabilidade de churn.
    """
    start = time.time()

    # Converter input para DataFrame
    import pandas as pd
    df_input = pd.DataFrame([customer.model_dump()])

    # Pré-processar
    X_transformed = preprocessor.transform(df_input)

    # Inferência
    X_tensor = torch.FloatTensor(X_transformed).to(device)
    with torch.no_grad():
        proba = model.predict_proba(X_tensor).item()

    # Classificar nível de risco
    if proba < 0.3:
        risk_level = "low"
    elif proba < 0.6:
        risk_level = "medium"
    else:
        risk_level = "high"

    latency_ms = (time.time() - start) * 1000

    logger.info(
        "Predição — prob=%.4f, risk=%s, latency=%.1fms",
        proba, risk_level, latency_ms,
    )

    return PredictionResponse(
        churn_probability=round(proba, 4),
        churn_prediction=int(proba >= THRESHOLD),
        risk_level=risk_level,
        latency_ms=round(latency_ms, 2),
    )