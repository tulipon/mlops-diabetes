"""Carga del modelo champion desde el Model Registry de MLflow."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import mlflow
import numpy as np
from mlflow import MlflowClient

from mlops.config import CHAMPION_ALIAS, MODEL_NAME, TRACKING_URI
from mlops.features import make_input_frame

logger = logging.getLogger(__name__)


@dataclass
class ChampionInfo:
    """Metadatos del modelo champion que se está sirviendo."""

    model_name: str
    version: str
    alias: str
    run_id: str | None = None
    feature_set: str | None = None
    model_family: str | None = None
    registered_at: str | None = None
    description: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)


class ChampionModel:
    """Envoltorio de inferencia que garantiza el orden de las features."""

    def __init__(self, model: Any, info: ChampionInfo) -> None:
        self.model = model
        self.info = info

    def predict(self, features: dict[str, float]) -> float:
        dataframe = make_input_frame(features)
        prediction = self.model.predict(dataframe)
        values = np.asarray(prediction).reshape(-1)
        if values.size == 0:
            raise RuntimeError("El modelo no devolvió ninguna predicción.")
        return float(values[0])


def _iso_timestamp(timestamp_ms: int | None) -> str | None:
    if not timestamp_ms:
        return None
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()


def load_champion(
    tracking_uri: str = TRACKING_URI,
    model_name: str = MODEL_NAME,
    alias: str = CHAMPION_ALIAS,
) -> ChampionModel:
    """Carga el modelo con alias ``champion`` y recupera sus metadatos.

    Raises:
        Exception: Si el registry no está disponible, el alias no existe o el
            artefacto no se puede cargar. La API captura el error y arranca en
            modo degradado.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    model_uri = f"models:/{model_name}@{alias}"

    logger.info("Cargando modelo champion: %s", model_uri)
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    version = client.get_model_version_by_alias(model_name, alias)

    metrics: dict[str, float] = {}
    feature_set: str | None = None
    model_family: str | None = None
    try:
        run = client.get_run(version.run_id)
        metrics = {
            key: float(value)
            for key, value in run.data.metrics.items()
            if key.startswith("test_") or key.startswith("cv_")
        }
        feature_set = run.data.tags.get("feature_set")
        model_family = run.data.tags.get("model_family")
    except Exception:  # pragma: no cover - depende de metadatos externos
        logger.exception("No se pudieron leer los metadatos del run %s", version.run_id)

    info = ChampionInfo(
        model_name=model_name,
        version=str(version.version),
        alias=alias,
        run_id=version.run_id,
        feature_set=feature_set,
        model_family=model_family,
        registered_at=_iso_timestamp(version.creation_timestamp),
        description=version.description,
        metrics=metrics,
    )
    logger.info("Modelo cargado: %s v%s", info.model_name, info.version)
    return ChampionModel(loaded_model, info)
