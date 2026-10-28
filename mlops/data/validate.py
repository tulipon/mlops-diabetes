"""Validación del contrato de datos antes de entrenar."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from mlops.config import (
    MIN_ROWS,
    RAW_DATA_PATH,
    REPORTS_DIR,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Error lanzado cuando el dataset no cumple el contrato de datos."""

    def __init__(self, report: "ValidationReport") -> None:
        self.report = report
        message = "Validación de datos fallida: " + "; ".join(report.errors)
        super().__init__(message)


@dataclass
class ValidationReport:
    """Resultado detallado de la validación."""

    rows: int
    columns: list[str]
    expected_columns: list[str]
    missing_columns: list[str] = field(default_factory=list)
    unexpected_columns: list[str] = field(default_factory=list)
    null_counts: dict[str, int] = field(default_factory=dict)
    non_numeric_columns: list[str] = field(default_factory=list)
    non_finite_columns: list[str] = field(default_factory=list)
    negative_target_rows: int = 0
    duplicate_rows: int = 0
    errors: list[str] = field(default_factory=list)
    passed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def validate_data(
    dataframe: pd.DataFrame,
    expected_columns: list[str] | None = None,
    min_rows: int = MIN_ROWS,
    allow_nulls: bool = False,
) -> ValidationReport:
    """Comprueba columnas, tipos, nulos, valores finitos y tamaño mínimo.

    Args:
        dataframe: Datos a validar.
        expected_columns: Contrato de columnas. Por defecto, el del proyecto.
        min_rows: Número mínimo de filas aceptado.
        allow_nulls: Si es ``True``, los nulos se registran pero no bloquean la
            validación. Se usa sobre los datos crudos porque la limpieza es la
            etapa responsable de tratarlos; tras limpiar se valida en estricto.

    Returns:
        Un ``ValidationReport`` con el detalle de las comprobaciones.

    Raises:
        DataValidationError: Si alguna comprobación crítica falla.
    """
    expected_columns = list(expected_columns or REQUIRED_COLUMNS)
    columns = [str(column) for column in dataframe.columns]
    errors: list[str] = []

    missing_columns = [column for column in expected_columns if column not in columns]
    unexpected_columns = [column for column in columns if column not in expected_columns]

    if missing_columns:
        errors.append(f"Faltan columnas obligatorias: {missing_columns}")
    if unexpected_columns:
        errors.append(f"Hay columnas no esperadas: {unexpected_columns}")
    if dataframe.empty:
        errors.append("El dataset está vacío.")
    if len(dataframe) < min_rows:
        errors.append(
            f"El dataset tiene {len(dataframe)} filas; el mínimo es {min_rows}."
        )

    existing_columns = [column for column in expected_columns if column in columns]
    null_counts = {
        column: int(dataframe[column].isna().sum())
        for column in existing_columns
    }
    columns_with_nulls = {
        column: count for column, count in null_counts.items() if count > 0
    }
    if columns_with_nulls:
        message = f"Hay valores nulos: {columns_with_nulls}"
        if allow_nulls:
            logger.warning("%s (se tratarán en la limpieza).", message)
        else:
            errors.append(message)

    non_numeric_columns = [
        column
        for column in existing_columns
        if not pd.api.types.is_numeric_dtype(dataframe[column])
    ]
    if non_numeric_columns:
        errors.append(f"Columnas no numéricas: {non_numeric_columns}")

    non_finite_columns: list[str] = []
    for column in existing_columns:
        if column in non_numeric_columns:
            continue
        values = dataframe[column].dropna().to_numpy(dtype=float)
        if values.size and not np.isfinite(values).all():
            non_finite_columns.append(column)
    if non_finite_columns:
        errors.append(f"Columnas con valores no finitos: {non_finite_columns}")

    negative_target_rows = 0
    if TARGET_COLUMN in columns and TARGET_COLUMN not in non_numeric_columns:
        negative_target_rows = int((dataframe[TARGET_COLUMN].dropna() < 0).sum())
        if negative_target_rows:
            errors.append(
                f"El target contiene {negative_target_rows} valores negativos."
            )

    report = ValidationReport(
        rows=int(len(dataframe)),
        columns=columns,
        expected_columns=expected_columns,
        missing_columns=missing_columns,
        unexpected_columns=unexpected_columns,
        null_counts=null_counts,
        non_numeric_columns=non_numeric_columns,
        non_finite_columns=non_finite_columns,
        negative_target_rows=negative_target_rows,
        duplicate_rows=int(dataframe.duplicated().sum()),
        errors=errors,
        passed=not errors,
    )

    if not report.passed:
        raise DataValidationError(report)

    logger.info(
        "Validación superada: %s filas, %s columnas, %s duplicados.",
        report.rows,
        len(report.columns),
        report.duplicate_rows,
    )
    return report


def write_validation_report(
    report: ValidationReport,
    output_path: str | Path = REPORTS_DIR / "validation_report.json",
) -> Path:
    """Persiste el informe de validación en JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    dataframe = pd.read_csv(RAW_DATA_PATH)
    try:
        report = validate_data(dataframe, allow_nulls=True)
    except DataValidationError as exc:
        write_validation_report(exc.report)
        print(f"Validación fallida: {exc}")
        raise SystemExit(1) from exc

    path = write_validation_report(report)
    print(f"Validación superada. Informe: {path}")


if __name__ == "__main__":
    main()
