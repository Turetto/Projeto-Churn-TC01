import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

# Colunas por tipo
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

CAT_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

DROP_COLS = ["customerID"]

# Coluna target
TARGET_COL = "Churn"


def load_raw_data(path: Path) -> pd.DataFrame:
    """Carrega o CSV bruto e aplica correções básicas de tipo."""
    logger.info("Carregando dados de %s", path)
    df = pd.read_csv(path)

    # Corrigir TotalCharges — vem como string com espaços
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Remover registros com TotalCharges nulo (clientes com tenure=0)
    n_before = len(df)
    df = df.dropna(subset=["TotalCharges"])
    n_after = len(df)
    logger.info("Removidos %d registros com TotalCharges nulo.", n_before - n_after)

    # Remover colunas sem valor preditivo
    df = df.drop(columns=DROP_COLS, errors="ignore")

    # Remover linhas duplicadas (novo)
    n_before = len(df)
    df = df.drop_duplicates()
    n_after = len(df)
    logger.info("Removidas %d linhas duplicadas.", n_before - n_after)

    logger.info("Dataset carregado: %d registros, %d colunas.", *df.shape)
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa features (X) e target (y)."""
    X = df.drop(columns=[TARGET_COL])
    y = (df[TARGET_COL] == "Yes").astype(int)
    logger.info("Target: %d positivos (%.1f%%)", y.sum(), y.mean() * 100)
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """
    Constrói o pipeline de pré-processamento.

    Numéricas: StandardScaler
    Categóricas: OneHotEncoder
    """
    numeric_transformer = Pipeline([
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUM_COLS),
            ("cat", categorical_transformer, CAT_COLS),
        ],
        remainder="drop",  # descarta colunas não listadas
        verbose_feature_names_out=False,
    )

    logger.info(
        "Preprocessor criado — %d numéricas, %d categóricas.",
        len(NUM_COLS),
        len(CAT_COLS),
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Retorna os nomes das features após o pré-processamento."""
    return list(preprocessor.get_feature_names_out())


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
    )

    RAW_PATH = Path("data/raw/telco_churn.csv")
    df = load_raw_data(RAW_PATH)
    X, y = split_features_target(df)

    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    print(f"\nShape após pré-processamento: {X_transformed.shape}")
    print(f"Features geradas: {get_feature_names(preprocessor)}")