import json
import logging
from pathlib import Path

import mlflow
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

DB_PATH = Path("models/mlflow.db").resolve()
MLFLOW_TRACKING_URI = f"sqlite:///{DB_PATH}"
CONFIG_PATH = Path("models/config.json")


def list_runs() -> pd.DataFrame:
    """Lista todos os runs do experimento churn-mlp."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    runs = mlflow.search_runs(
        experiment_names=["churn-mlp"],
        order_by=["metrics.roc_auc DESC"],
    )

    cols = ["run_id", "tags.mlflow.runName", "metrics.roc_auc",
            "metrics.recall", "metrics.f1", "metrics.pr_auc",
            "params.threshold", "params.hidden_dims"]

    df = runs[cols].copy()
    df.columns = ["run_id", "run_name", "roc_auc",
                  "recall", "f1", "pr_auc", "threshold", "hidden_dims"]
    df["run_id_short"] = df["run_id"].str[:8]

    return df


def save_config(run_id: str, threshold: float) -> None:
    """Salva a configuração do modelo selecionado."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    run = mlflow.get_run(run_id)
    config = {
        "run_id":    run_id,
        "run_name":  run.info.run_name,
        "threshold": threshold,
        "roc_auc":   run.data.metrics.get("roc_auc"),
        "recall":    run.data.metrics.get("recall"),
        "f1":        run.data.metrics.get("f1"),
        "pr_auc":    run.data.metrics.get("pr_auc"),
    }

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    logger.info("Configuração salva em %s", CONFIG_PATH)
    logger.info("  run_id:    %s", run_id)
    logger.info("  run_name:  %s", config["run_name"])
    logger.info("  threshold: %.2f", threshold)
    logger.info("  roc_auc:   %.4f", config["roc_auc"])
    logger.info("  recall:    %.4f", config["recall"])


def load_config() -> dict:
    """Carrega a configuração do modelo ativo."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Config não encontrado em {CONFIG_PATH}. "
            "Execute: uv run python src/model_selector.py"
        )

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    logger.info("Configuração carregada: %s (threshold=%.2f)",
                config["run_name"], config["threshold"])
    return config


if __name__ == "__main__":
    # Listar runs disponíveis
    df = list_runs()

    print("\n=== Runs disponíveis no MLflow ===")
    print(df[["run_id_short", "run_name", "roc_auc",
              "recall", "f1", "threshold"]].to_string(index=True))

    # Selecionar run interativamente
    print("\nDigite o índice do run que deseja usar em produção:")
    idx = int(input("> "))
    selected = df.iloc[idx]

    print(f"\nRun selecionado: {selected['run_name']}")
    print(f"  ROC-AUC:  {selected['roc_auc']:.4f}")
    print(f"  Recall:   {selected['recall']:.4f}")
    print(f"  Threshold atual no MLflow: {selected['threshold']}")

    print("\nDigite o threshold desejado (Enter para usar o do MLflow):")
    threshold_input = input("> ").strip()

    if threshold_input == "":
        threshold = float(selected["threshold"]) if selected["threshold"] else 0.5
    else:
        threshold = float(threshold_input)

    save_config(run_id=selected["run_id"], threshold=threshold)
    print(f"\nModelo {selected['run_name']} configurado com threshold={threshold:.2f}")
    print("Reinicie a API para aplicar as mudanças.")