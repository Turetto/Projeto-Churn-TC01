import logging
import zipfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


RAW_DATA_DIR = Path("data/raw")
DATASET_SLUG = "blastchar/telco-customer-churn"
FINAL_FILENAME = "telco_churn.csv"
ORIGINAL_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def download_dataset(
    dataset_slug: str = DATASET_SLUG,
    output_dir: Path = RAW_DATA_DIR,
    final_filename: str = FINAL_FILENAME,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / final_filename

    if final_path.exists():
        logger.info("Dataset já existe em %s — pulando download.", final_path)
        return final_path

    logger.info("Iniciando download do dataset '%s'...", dataset_slug)

    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            dataset=dataset_slug,
            path=str(output_dir),
            unzip=False,
        )
        logger.info("Download concluído.")
    except Exception as e:
        logger.error("Falha no download: %s", e)
        raise

    zip_path = output_dir / f"{dataset_slug.split('/')[-1]}.zip"
    if zip_path.exists():
        logger.info("Extraindo %s...", zip_path.name)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_dir)
        zip_path.unlink()
        logger.info("Zip removido após extração.")

    original_path = output_dir / ORIGINAL_FILENAME
    if original_path.exists() and not final_path.exists():
        original_path.rename(final_path)
        logger.info("Arquivo renomeado para '%s'.", final_filename)

    if not final_path.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em {final_path} após o download."
        )

    logger.info("Dataset disponível em: %s", final_path)
    return final_path


if __name__ == "__main__":
    download_dataset()