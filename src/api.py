from pathlib import Path
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from src.features import build_preprocessor, load_raw_data, split_features_target
from src.train import get_device, set_seeds
from src.model_selector import load_config
import logging
import time
import torch
import mlflow.pytorch

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

@app.on_event("startup")
async def load_model() -> None:
    global model, preprocessor, device, threshold

    logger.info("Iniciando carregamento do modelo...")
    set_seeds(SEED)
    device = get_device()

    # Carregar configuração do modelo ativo
    config = load_config()
    threshold = config["threshold"]
    logger.info("Modelo: %s — threshold: %.2f", config["run_name"], threshold)

    # Carregar preprocessor
    try:
        df = load_raw_data(Path("data/raw/telco_churn.csv"))
        X, _ = split_features_target(df)
        preprocessor = build_preprocessor()
        preprocessor.fit(X)
        logger.info("Preprocessor carregado.")
    except Exception as e:
        logger.error("Erro ao carregar preprocessor: %s", e)
        raise

    # Carregar modelo do run configurado
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"runs:/{config['run_id']}/model"
        model = mlflow.pytorch.load_model(model_uri, map_location=device)
        model.eval()
        logger.info("Modelo carregado: %s", config["run_name"])
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
    start = time.time()

    import pandas as pd
    df_input = pd.DataFrame([customer.model_dump()])
    X_transformed = preprocessor.transform(df_input)

    X_tensor = torch.FloatTensor(X_transformed).to(device)
    with torch.no_grad():
        proba = model.predict_proba(X_tensor).item()

    # Usar threshold carregado da configuração
    churn_pred = int(proba >= threshold)

    if proba < 0.3:
        risk_level = "low"
    elif proba < 0.6:
        risk_level = "medium"
    else:
        risk_level = "high"

    latency_ms = (time.time() - start) * 1000

    logger.info(
        "Predição — prob=%.4f, threshold=%.2f, pred=%d, risk=%s, latency=%.1fms",
        proba, threshold, churn_pred, risk_level, latency_ms,
    )

    return PredictionResponse(
        churn_probability=round(proba, 4),
        churn_prediction=churn_pred,
        risk_level=risk_level,
        latency_ms=round(latency_ms, 2),
    )

@app.get("/model-info", tags=["monitoring"]) # novo
async def model_info() -> dict:
    """Retorna informações sobre o modelo ativo."""
    from src.model_selector import load_config
    config = load_config()
    return {
        "run_name":  config["run_name"],
        "run_id":    config["run_id"],
        "threshold": config["threshold"],
        "roc_auc":   config["roc_auc"],
        "recall":    config["recall"],
    }