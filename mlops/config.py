"""Configuración central del proyecto.

Todos los valores se pueden sobrescribir con variables de entorno, lo que
permite ejecutar el pipeline y la API con un tracking URI remoto sin cambiar
el código.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if not value:
        return default
    return Path(value).expanduser().resolve()


# Rutas del proyecto
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = _env_path("RAW_DATA_PATH", DATA_DIR / "raw" / "diabetes.csv")
PROCESSED_DATA_PATH = _env_path(
    "PROCESSED_DATA_PATH", DATA_DIR / "processed" / "diabetes_clean.csv"
)
REPORTS_DIR = _env_path("REPORTS_DIR", PROJECT_ROOT / "reports")

# MLflow
MLFLOW_DB_PATH = _env_path("MLFLOW_DB_PATH", PROJECT_ROOT / "mlflow.db")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB_PATH}")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "diabetes-progression")
MODEL_NAME = os.getenv("MODEL_NAME", "diabetes-progression-model")
CHAMPION_ALIAS = os.getenv("CHAMPION_ALIAS", "champion")
MODEL_ARTIFACT_NAME = "model"

# Contrato de datos
TARGET_COLUMN = "target"
FEATURE_COLUMNS = [
    "age",
    "sex",
    "bmi",
    "bp",
    "s1",
    "s2",
    "s3",
    "s4",
    "s5",
    "s6",
]
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]
MIN_ROWS = int(os.getenv("MIN_ROWS", "300"))

# Entrenamiento
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
CV_FOLDS = int(os.getenv("CV_FOLDS", "5"))

# Calidad y promoción del champion. El umbral es deliberadamente estricto:
# el baseline exploratorio del notebook ronda RMSE 53.8-53.9.
RMSE_THRESHOLD = float(os.getenv("RMSE_THRESHOLD", "54.0"))
MIN_R2 = float(os.getenv("MIN_R2", "0.40"))
