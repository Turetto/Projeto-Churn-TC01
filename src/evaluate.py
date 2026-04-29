from sklearn.model_selection import train_test_split
from src.features import load_raw_data, split_features_target, build_preprocessor
from src.train import train_model, get_device, set_seeds
from src.model import ChurnMLP
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
import numpy as np
import matplotlib.pyplot as plt
import logging
import torch

logger = logging.getLogger(__name__)


def get_predictions(
    model: ChurnMLP,
    X: np.ndarray,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retorna probabilidades e classes preditas.

    Returns:
        proba: probabilidades de churn [0, 1]
        preds: classes binárias (0 ou 1) usando threshold
    """
    X_tensor = torch.FloatTensor(X).to(device)
    proba = model.predict_proba(X_tensor).cpu().numpy().flatten()
    preds = (proba >= threshold).astype(int)
    return proba, preds


def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Calculo das métricas.

    Returns:
        dicionário com todas as métricas
    """
    metrics = {
        "roc_auc":   roc_auc_score(y_true, y_proba),
        "pr_auc":    average_precision_score(y_true, y_proba),
        "f1":        f1_score(y_true, y_pred),
        "recall":    recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
    }

    logger.info("=== Métricas de Avaliação ===")
    for name, value in metrics.items():
        logger.info("  %-12s: %.4f", name, value)

    return metrics


def compute_cost_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_fn: float = 500.0,
    cost_fp: float = 50.0,
) -> dict:
    """
    Análise de custo de negócio — trade-off FP vs FN.

    Args:
        cost_fn: custo de um falso negativo (churn não detectado = perda do MRR)
        cost_fp: custo de um falso positivo (campanha de retenção desnecessária)

    Returns:
        dicionário com custos e economia estimada
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    total_cost = (fn * cost_fn) + (fp * cost_fp)
    # Custo baseline — se não houvesse modelo (todos os churns perdidos)
    baseline_cost = y_true.sum() * cost_fn
    savings = baseline_cost - total_cost
    savings_pct = (savings / baseline_cost) * 100 if baseline_cost > 0 else 0

    analysis = {
        "true_positives":  int(tp),
        "true_negatives":  int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "cost_fn_unit":    cost_fn,
        "cost_fp_unit":    cost_fp,
        "total_cost":      total_cost,
        "baseline_cost":   baseline_cost,
        "savings":         savings,
        "savings_pct":     savings_pct,
    }

    logger.info("=== Análise de Custo de Negócio ===")
    logger.info("  Falsos Negativos (churns perdidos): %d × R$%.0f = R$%.0f",
                fn, cost_fn, fn * cost_fn)
    logger.info("  Falsos Positivos (campanhas extras): %d × R$%.0f = R$%.0f",
                fp, cost_fp, fp * cost_fp)
    logger.info("  Custo com modelo:   R$%.0f", total_cost)
    logger.info("  Custo sem modelo:   R$%.0f", baseline_cost)
    logger.info("  Economia estimada:  R$%.0f (%.1f%%)", savings, savings_pct)

    return analysis


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """Plota a curva ROC."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Curva ROC — ChurnMLP")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Curva ROC salva em %s", save_path)

    plt.show()


def plot_pr_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """Plota a curva Precision-Recall."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color="tomato", lw=2,
            label=f"PR curve (AUC = {pr_auc:.4f})")
    ax.axhline(y=baseline, color="gray", linestyle="--", lw=1,
               label=f"Baseline ({baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curva Precision-Recall — ChurnMLP")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Curva PR salva em %s", save_path)

    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """Plota a matriz de confusão."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Não Churn (0)", "Churn (1)"])
    ax.set_yticklabels(["Não Churn (0)", "Churn (1)"])
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão — ChurnMLP")

    # Valores dentro da matriz
    for i, j, val in [(0,0,tn),(0,1,fp),(1,0,fn),(1,1,tp)]:
        ax.text(j, i, str(val), ha="center", va="center",
                color="white" if val > cm.max()/2 else "black",
                fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Matriz de confusão salva em %s", save_path)

    plt.show()


def plot_training_history(
    history: dict,
    save_path: Path | None = None,
) -> None:
    """Plota as curvas de loss de treino e validação."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], label="Train loss",
            color="steelblue", lw=2)
    ax.plot(epochs, history["val_loss"], label="Val loss",
            color="tomato", lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.set_title("Curvas de Aprendizado — ChurnMLP")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Histórico de treino salvo em %s", save_path)

    plt.show()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
    )
 
    SEED = 1312
    set_seeds(SEED)
    device = get_device()

    # Carregar e preparar dados
    df = load_raw_data(Path("data/raw/telco_churn.csv"))
    X, y = split_features_target(df)

    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)
    y_array = y.values

    X_train, X_val, y_train, y_val = train_test_split(
        X_transformed, y_array,
        test_size=0.2,
        random_state=SEED,
        stratify=y_array,
    )

    # Treinar modelo
    model, history = train_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        input_dim=X_train.shape[1],
    )

    # Avaliação completa
    y_proba, y_pred = get_predictions(model, X_val, device)

    metrics = compute_metrics(y_val, y_proba, y_pred)
    cost = compute_cost_analysis(y_val, y_pred)

    # Criar pasta para os gráficos
    FIGURES_DIR = Path("docs/figures")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Gráficos 
    plot_training_history(history, save_path=FIGURES_DIR / "training_history.png")
    plot_roc_curve(y_val, y_proba, save_path=FIGURES_DIR / "roc_curve.png")
    plot_pr_curve(y_val, y_proba, save_path=FIGURES_DIR / "pr_curve.png")
    plot_confusion_matrix(y_val, y_pred, save_path=FIGURES_DIR / "confusion_matrix.png")