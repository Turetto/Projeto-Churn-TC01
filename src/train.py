from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from src.features import load_raw_data, split_features_target, build_preprocessor
from src.model import ChurnMLP, build_model
import numpy as np
import torch.nn as nn
import logging
import random
import torch

logger = logging.getLogger(__name__)

SEED = 1312

def set_seeds(seed: int = SEED) -> None:
    """Fixa seeds em todas as bibliotecas para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Avalia utilização de GPU se disponível"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Dispositivo de treino: %s", device)
    return device


def make_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader]:
    """
    Converte arrays numpy em DataLoaders do PyTorch para batching e embaralhamento automaticamente.
    """
    # Converter para tensores PyTorch
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)  # correçao
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1)

    # Criar datasets
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)

    # Criar dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=lambda _: np.random.seed(SEED),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    logger.info(
        "DataLoaders criados — treino: %d batches, validação: %d batches.",
        len(train_loader),
        len(val_loader),
    )
    return train_loader, val_loader


class EarlyStopping:
    """
    Parar o treino quando a métrica de validação para de melhorar.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-5) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """
        Atualiza o estado do early stopping.
        Retorna True se o treino deve parar.
        """
        if val_loss < self.best_loss - self.min_delta:
            # Houve melhora — resetar contador
            self.best_loss = val_loss
            self.counter = 0
        else:
            # Sem melhora — incrementar contador
            self.counter += 1
            logger.info(
                "EarlyStopping: sem melhora por %d/%d epochs.",
                self.counter,
                self.patience,
            )
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info("EarlyStopping ativado — encerrando treino.")

        return self.should_stop


def train_one_epoch(
    model: ChurnMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Treina o modelo por uma epoca. Retorna o loss médio."""
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Zerar gradientes acumulados
        optimizer.zero_grad()

        # calcular predições
        predictions = model(X_batch)

        # Calcular loss
        loss = criterion(predictions, y_batch)

        # calcular gradientes
        loss.backward()

        # Atualizar pesos
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(
    model: ChurnMLP,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Avalia o modelo na validação. Retorna o loss médio."""
    model.eval()  # desativa Dropout
    total_loss = 0.0

    with torch.no_grad():  # sem cálculo de gradientes
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            total_loss += loss.item()

    return total_loss / len(loader)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    hidden_dims: list[int] = [64, 32, 16],
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    max_epochs: int = 100,
    patience: int = 10,
) -> tuple[ChurnMLP, dict]:
    """
    Função principal de treino.
    Retorna o modelo treinado e o histórico de métricas.
    """
    set_seeds(SEED)
    device = get_device()

    # Criar dataloaders
    train_loader, val_loader = make_dataloaders(
        X_train, y_train, X_val, y_val, batch_size
    )

    # Criar modelo
    model = build_model(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
    )
    model = model.to(device)

    # Loss function e otimizador
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping
    early_stopping = EarlyStopping(patience=patience)

    # Histórico de métricas
    history = {
        "train_loss": [],
        "val_loss": [],
    }

    logger.info("Iniciando treino — max_epochs=%d, patience=%d", max_epochs, patience)

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        logger.info(
            "Epoch %3d/%d — train_loss: %.4f — val_loss: %.4f",
            epoch, max_epochs, train_loss, val_loss,
        )

        if early_stopping.step(val_loss):
            break

    logger.info("Treino concluído em %d epochs.", epoch)
    return model, history


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
    )

    # Carregar dados
    df = load_raw_data(Path("data/raw/telco_churn.csv"))
    X, y = split_features_target(df)

    # Pré-processar
    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)
    y_array = y.values

    # Split treino/validação
    X_train, X_val, y_train, y_val = train_test_split(
        X_transformed, y_array,
        test_size=0.2,
        random_state=SEED,
        stratify=y_array,
    )

    # Treinar
    model, history = train_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        input_dim=X_train.shape[1],
    )

    print(f"\nEpochs treinadas: {len(history['train_loss'])}")
    print(f"Melhor val_loss: {min(history['val_loss']):.4f}")