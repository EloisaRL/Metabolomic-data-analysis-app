from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent

UPLOAD_FOLDER = APP_ROOT / "pre-processed-datasets"

CACHE_DIR = APP_ROOT / "cache"
PROJECTS_DIR = APP_ROOT / "Projects"
PREPROCESSED_DIR = APP_ROOT / "pre-processed-datasets"
DATA_FLOWS_DIR = APP_ROOT / "data_preprocessing_flows"
