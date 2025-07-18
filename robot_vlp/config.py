from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
EXPERIMENT_DATA_DIR = RAW_DATA_DIR /'experiments'

MODELS_DIR = PROJ_ROOT / "models"
VLP_MODELS_DIR = MODELS_DIR / "vlp"
TRAINING_LOGS_DIR = MODELS_DIR / "training_logs"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "67c4db6d4bd74c401fb0585d/figures"
TABLES_DIR = REPORTS_DIR / "tables"

RESULTS_FILE = REPORTS_DIR / "67c4db6d4bd74c401fb0585d/results.tex"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
