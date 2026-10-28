"""Tests de la ingeniería de features compartida."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from mlops.features import (
    ENGINEERED_COLUMNS,
    FEATURE_SETS,
    RAW_FEATURES,
    add_engineered_features,
    build_model_pipeline,
    feature_columns_for,
    make_input_frame,
)

SAMPLE = {
    "age": 0.038076,
    "sex": 0.050680,
    "bmi": 0.061696,
    "bp": 0.021872,
    "s1": -0.044223,
    "s2": -0.034821,
    "s3": -0.043401,
    "s4": -0.002592,
    "s5": 0.019907,
    "s6": -0.017646,
}


def test_add_engineered_features_adds_expected_columns() -> None:
    dataframe = pd.DataFrame([SAMPLE])
    result = add_engineered_features(dataframe)

    assert set(result.columns) == set(SAMPLE) | set(ENGINEERED_COLUMNS)
    expected_bmi_age = SAMPLE["bmi"] * SAMPLE["age"]
    expected_bp_bmi = SAMPLE["bp"] * SAMPLE["bmi"]
    expected_ratio = SAMPLE["s1"] / SAMPLE["s2"]

    assert np.isclose(result.loc[0, "bmi_age_interaction"], expected_bmi_age)
    assert np.isclose(result.loc[0, "bp_bmi_interaction"], expected_bp_bmi)
    assert np.isclose(result.loc[0, "s1_s2_ratio"], expected_ratio)


def test_add_engineered_features_protects_division_by_zero() -> None:
    dataframe = pd.DataFrame([{**SAMPLE, "s2": 0.0}])
    result = add_engineered_features(dataframe)

    assert np.isfinite(result.loc[0, "s1_s2_ratio"])


def test_feature_columns_for_returns_raw_and_engineered_sets() -> None:
    assert feature_columns_for(RAW_FEATURES) == list(SAMPLE)
    assert feature_columns_for("engineered") == list(SAMPLE) + ENGINEERED_COLUMNS


def test_feature_columns_for_rejects_unknown_set() -> None:
    with pytest.raises(ValueError):
        feature_columns_for("unknown")


def test_build_model_pipeline_rejects_unknown_set() -> None:
    with pytest.raises(ValueError):
        build_model_pipeline(LinearRegression(), "unknown")


@pytest.mark.parametrize("feature_set", FEATURE_SETS)
def test_pipeline_fits_and_predicts(feature_set: str) -> None:
    rows = [
        {**SAMPLE, "age": SAMPLE["age"] + offset, "bmi": SAMPLE["bmi"] + offset}
        for offset in np.linspace(-0.01, 0.01, 30)
    ]
    dataframe = pd.DataFrame(rows)
    target = np.linspace(100, 300, len(dataframe))

    pipeline = build_model_pipeline(LinearRegression(), feature_set)
    pipeline.fit(dataframe, target)
    predictions = pipeline.predict(dataframe.head(3))

    assert predictions.shape == (3,)
    assert np.isfinite(predictions).all()


def test_make_input_frame_orders_columns() -> None:
    dataframe = make_input_frame(dict(reversed(list(SAMPLE.items()))))
    assert list(dataframe.columns) == list(SAMPLE)
