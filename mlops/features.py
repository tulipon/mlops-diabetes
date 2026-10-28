"""Ingeniería de features compartida por entrenamiento y serving.

Es crucial que el transformador usado al entrenar sea exactamente el mismo que
se aplica en producción. Por eso el modelo se registra como un ``Pipeline`` de
scikit-learn que recibe las 10 variables originales y aplica, si corresponde,
las interacciones exploradas en el notebook.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from mlops.config import FEATURE_COLUMNS

RAW_FEATURES = "raw"
ENGINEERED_FEATURES = "engineered"
FEATURE_SETS = (RAW_FEATURES, ENGINEERED_FEATURES)

ENGINEERED_COLUMNS = [
    "bmi_age_interaction",
    "bp_bmi_interaction",
    "s1_s2_ratio",
]


def add_engineered_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Añade las interacciones exploradas en el notebook.

    La división ``s1 / s2`` se protege con un denominador mínimo para evitar
    infinitos cuando ``s2`` es prácticamente cero.
    """
    if not isinstance(dataframe, pd.DataFrame):
        dataframe = pd.DataFrame(dataframe, columns=FEATURE_COLUMNS)

    result = dataframe.copy()
    result["bmi_age_interaction"] = result["bmi"] * result["age"]
    result["bp_bmi_interaction"] = result["bp"] * result["bmi"]

    denominator = result["s2"].astype(float)
    denominator = denominator.mask(denominator.abs() < 1e-9, 1e-9)
    result["s1_s2_ratio"] = result["s1"] / denominator
    return result


def feature_columns_for(feature_set: str) -> list[str]:
    """Devuelve las columnas de entrada que espera el modelo."""
    if feature_set == RAW_FEATURES:
        return list(FEATURE_COLUMNS)
    if feature_set == ENGINEERED_FEATURES:
        return list(FEATURE_COLUMNS) + list(ENGINEERED_COLUMNS)
    raise ValueError(
        f"Feature set desconocido: {feature_set!r}. "
        f"Opciones válidas: {FEATURE_SETS}"
    )


def build_model_pipeline(estimator: BaseEstimator, feature_set: str) -> Pipeline:
    """Construye el pipeline que se entrena y se registra en MLflow.

    Args:
        estimator: Modelo de scikit-learn.
        feature_set: ``raw`` o ``engineered``.

    Returns:
        Pipeline que acepta el DataFrame original de 10 columnas.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Feature set desconocido: {feature_set!r}. "
            f"Opciones válidas: {FEATURE_SETS}"
        )

    steps: list[tuple[str, object]] = []
    if feature_set == ENGINEERED_FEATURES:
        steps.append(
            (
                "features",
                FunctionTransformer(add_engineered_features, validate=False),
            )
        )
    steps.append(("model", estimator))
    return Pipeline(steps)


def make_input_frame(values: dict[str, float]) -> pd.DataFrame:
    """Construye un DataFrame de una fila con el orden de columnas esperado."""
    missing = [column for column in FEATURE_COLUMNS if column not in values]
    if missing:
        raise ValueError(f"Faltan features: {missing}")
    return pd.DataFrame([{column: values[column] for column in FEATURE_COLUMNS}])
