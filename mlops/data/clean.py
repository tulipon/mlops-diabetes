"""Limpieza básica del dataset antes del entrenamiento."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from mlops.config import (
    MIN_ROWS,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    REPORTS_DIR,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)


class DataCleaningError(ValueError):
    """Error lanzado cuando la limpieza deja un dataset no utilizable."""


@dataclass
class CleaningStats:
    """Trazabilidad de las acciones aplicadas durante la limpieza."""

    input_rows: int
    output_rows: int
    dropped_duplicates: int = 0
    dropped_missing: int = 0
    values_coerced_to_null: int = 0
    dropped_negative_target: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def clean_data(
    dataframe: pd.DataFrame,
    min_rows: int = MIN_ROWS,
) -> tuple[pd.DataFrame, CleaningStats]:
    """Elimina duplicados y nulos, fuerza tipos numéricos y valida el target.

    La limpieza es deliberadamente simple y trazable: cada decisión queda
    registrada en ``CleaningStats`` para poder auditar el pipeline.

    Args:
        dataframe: Datos de entrada.
        min_rows: Mínimo de filas que debe conservar el resultado.

    Returns:
        Una tupla ``(dataframe_limpio, estadísticas)``.

    Raises:
        DataCleaningError: Si faltan columnas o el resultado es demasiado pequeño.
    """
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in dataframe.columns
    ]
    if missing_columns:
        raise DataCleaningError(
            f"No se puede limpiar el dataset; faltan columnas: {missing_columns}"
        )

    working = dataframe.copy()
    input_rows = len(working)

    before_duplicates = len(working)
    working = working.drop_duplicates()
    dropped_duplicates = before_duplicates - len(working)

    # Conversión defensiva: un valor no numérico pasa a nulo y se elimina.
    values_coerced_to_null = 0
    for column in REQUIRED_COLUMNS:
        before_notna = working[column].notna()
        working[column] = pd.to_numeric(working[column], errors="coerce")
        values_coerced_to_null += int((before_notna & working[column].isna()).sum())

    before_missing = len(working)
    working = working.dropna(subset=REQUIRED_COLUMNS)
    dropped_missing = before_missing - len(working)

    negative_target = working[TARGET_COLUMN] < 0
    dropped_negative_target = int(negative_target.sum())
    working = working.loc[~negative_target]

    working = working.reset_index(drop=True)
    stats = CleaningStats(
        input_rows=input_rows,
        output_rows=len(working),
        dropped_duplicates=dropped_duplicates,
        dropped_missing=dropped_missing,
        values_coerced_to_null=values_coerced_to_null,
        dropped_negative_target=dropped_negative_target,
    )

    if len(working) < min_rows:
        raise DataCleaningError(
            f"Tras limpiar quedan {len(working)} filas; el mínimo es {min_rows}. "
            f"Estadísticas: {stats.to_dict()}"
        )

    logger.info(
        "Limpieza completada: %s -> %s filas (duplicados=%s, nulos=%s).",
        stats.input_rows,
        stats.output_rows,
        stats.dropped_duplicates,
        stats.dropped_missing,
    )
    return working, stats


def write_cleaning_report(
    stats: CleaningStats,
    output_path: str | Path = REPORTS_DIR / "cleaning_report.json",
) -> Path:
    """Persiste las estadísticas de limpieza en JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(stats.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def clean_file(
    input_path: str | Path = RAW_DATA_PATH,
    output_path: str | Path = PROCESSED_DATA_PATH,
    report_path: str | Path = REPORTS_DIR / "cleaning_report.json",
) -> tuple[Path, CleaningStats]:
    """Limpia un CSV y guarda el resultado y su informe."""
    dataframe = pd.read_csv(input_path)
    cleaned, stats = clean_data(dataframe)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    write_cleaning_report(stats, report_path)
    return output_path, stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        path, stats = clean_file()
    except DataCleaningError as exc:
        print(f"Limpieza fallida: {exc}")
        raise SystemExit(1) from exc

    print(f"Datos limpios guardados en: {path}")
    print(json.dumps(stats.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
