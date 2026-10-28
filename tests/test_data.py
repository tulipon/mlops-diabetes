"""Tests de extracción, validación y limpieza de datos."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlops.config import REQUIRED_COLUMNS, TARGET_COLUMN
from mlops.data.clean import DataCleaningError, clean_data
from mlops.data.extract import extract_data, sha256_file
from mlops.data.validate import DataValidationError, validate_data


@pytest.fixture(scope="module")
def raw_dataframe(tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    path = tmp_path_factory.mktemp("data") / "diabetes.csv"
    extract_data(path)
    return pd.read_csv(path)


def test_extract_data_creates_expected_schema(raw_dataframe: pd.DataFrame) -> None:
    assert raw_dataframe.shape == (442, 11)
    assert list(raw_dataframe.columns) == REQUIRED_COLUMNS
    assert raw_dataframe[TARGET_COLUMN].notna().all()


def test_sha256_file_returns_64_hex_characters(tmp_path) -> None:
    path = tmp_path / "file.txt"
    path.write_text("mlops", encoding="utf-8")
    digest = sha256_file(path)
    assert len(digest) == 64
    assert all(character in "0123456789abcdef" for character in digest)


def test_validate_data_accepts_valid_dataset(raw_dataframe: pd.DataFrame) -> None:
    report = validate_data(raw_dataframe)
    assert report.passed is True
    assert report.errors == []
    assert report.duplicate_rows == 0


def test_validate_data_allows_nulls_only_when_requested(
    raw_dataframe: pd.DataFrame,
) -> None:
    dataframe = raw_dataframe.copy()
    dataframe.loc[0, "bmi"] = np.nan

    permissive = validate_data(dataframe, allow_nulls=True)
    assert permissive.passed is True
    assert permissive.null_counts["bmi"] == 1

    with pytest.raises(DataValidationError) as error:
        validate_data(dataframe)
    assert "valores nulos" in str(error.value)


def test_validate_data_rejects_missing_column(raw_dataframe: pd.DataFrame) -> None:
    dataframe = raw_dataframe.drop(columns=["bmi"])
    with pytest.raises(DataValidationError) as error:
        validate_data(dataframe)
    assert "Faltan columnas obligatorias" in str(error.value)
    assert "bmi" in str(error.value)


def test_validate_data_rejects_non_numeric_column(
    raw_dataframe: pd.DataFrame,
) -> None:
    dataframe = raw_dataframe.copy()
    dataframe["age"] = "texto"
    with pytest.raises(DataValidationError) as error:
        validate_data(dataframe)
    assert "Columnas no numéricas" in str(error.value)


def test_validate_data_rejects_too_few_rows(raw_dataframe: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError):
        validate_data(raw_dataframe.head(10), min_rows=300)


def test_clean_data_removes_duplicates_and_nulls(
    raw_dataframe: pd.DataFrame,
) -> None:
    dataframe = raw_dataframe.head(20).copy()
    dataframe.loc[0, "bmi"] = np.nan
    dataframe = pd.concat(
        [dataframe, dataframe.iloc[[1]]], ignore_index=True
    )

    cleaned, stats = clean_data(dataframe, min_rows=1)

    assert len(cleaned) == 19
    assert stats.input_rows == 21
    assert stats.dropped_duplicates == 1
    assert stats.dropped_missing == 1
    assert cleaned.isna().sum().sum() == 0


def test_clean_data_drops_negative_target(raw_dataframe: pd.DataFrame) -> None:
    dataframe = raw_dataframe.head(20).copy()
    dataframe.loc[0, TARGET_COLUMN] = -10

    cleaned, stats = clean_data(dataframe, min_rows=1)

    assert len(cleaned) == 19
    assert stats.dropped_negative_target == 1
    assert (cleaned[TARGET_COLUMN] >= 0).all()


def test_clean_data_rejects_insufficient_rows(
    raw_dataframe: pd.DataFrame,
) -> None:
    with pytest.raises(DataCleaningError):
        clean_data(raw_dataframe.head(10), min_rows=300)
