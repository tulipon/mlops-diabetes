"""Extracción del dataset de diabetes de scikit-learn a un CSV crudo."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_diabetes

from mlops.config import RAW_DATA_PATH, REQUIRED_COLUMNS

logger = logging.getLogger(__name__)


def sha256_file(path: str | Path) -> str:
    """Calcula el SHA-256 de un fichero para versionar los datos."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_data(output_path: str | Path = RAW_DATA_PATH) -> Path:
    """Carga el dataset de scikit-learn y lo persiste como CSV crudo.

    Args:
        output_path: Ruta del CSV de salida.

    Returns:
        La ruta del fichero escrito.
    """
    output_path = Path(output_path)
    dataset = load_diabetes(as_frame=True)
    dataframe = dataset.frame.copy()

    missing = set(REQUIRED_COLUMNS) - set(dataframe.columns)
    if missing:
        raise RuntimeError(
            "El dataset de scikit-learn no contiene las columnas esperadas: "
            f"{sorted(missing)}"
        )

    dataframe = dataframe[REQUIRED_COLUMNS]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)

    logger.info(
        "Dataset extraído: %s filas x %s columnas -> %s",
        dataframe.shape[0],
        dataframe.shape[1],
        output_path,
    )
    return output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    path = extract_data()
    print(f"Datos crudos guardados en: {path}")
    print(f"SHA-256: {sha256_file(path)}")


if __name__ == "__main__":
    main()
