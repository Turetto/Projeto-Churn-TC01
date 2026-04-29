from src.model import build_model
from src.features import build_preprocessor, load_raw_data, split_features_target
from src.train import set_seeds, get_device
from pathlib import Path
import torch

SEED = 1312
INPUT_DIM = 44


def test_model_instantiation():
    """Modelo deve instanciar sem erros."""
    set_seeds(SEED)
    model = build_model(input_dim=INPUT_DIM)
    assert model is not None


def test_model_forward_pass():
    """Forward pass deve retornar tensor com shape correto."""
    set_seeds(SEED)
    device = get_device()
    model = build_model(input_dim=INPUT_DIM).to(device)

    x = torch.randn(8, INPUT_DIM).to(device)
    output = model(x)

    assert output.shape == (8, 1)


def test_model_output_range():
    """Saída do modelo deve estar entre 0 e 1 (sigmoid)."""
    set_seeds(SEED)
    device = get_device()
    model = build_model(input_dim=INPUT_DIM).to(device)

    x = torch.randn(32, INPUT_DIM).to(device)
    output = model(x)

    assert output.min().item() >= 0.0
    assert output.max().item() <= 1.0


def test_preprocessor_output_shape():
    """Preprocessor deve gerar 44 features."""
    df = load_raw_data(Path("data/raw/telco_churn.csv"))
    X, _ = split_features_target(df)

    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    assert X_transformed.shape[1] == INPUT_DIM


def test_predict_proba_no_gradient():
    """predict_proba não deve calcular gradientes."""
    set_seeds(SEED)
    device = get_device()
    model = build_model(input_dim=INPUT_DIM).to(device)

    x = torch.randn(4, INPUT_DIM).to(device)
    proba = model.predict_proba(x)

    assert not proba.requires_grad