"""Registro del modelo en MLflow y selección del champion."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from mlops.config import (
    CHAMPION_ALIAS,
    MIN_R2,
    MODEL_NAME,
    RMSE_THRESHOLD,
    TRACKING_URI,
)
from mlops.training.train import TrainingResult

logger = logging.getLogger(__name__)


@dataclass
class PromotionDecision:
    """Decisión de promoción y su justificación."""

    promoted: bool
    reason: str
    version: str
    alias: str
    previous_version: str | None = None
    previous_rmse: float | None = None
    test_rmse: float | None = None
    test_r2: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_promotion(
    test_rmse: float,
    test_r2: float,
    current_champion_rmse: float | None = None,
    rmse_threshold: float = RMSE_THRESHOLD,
    min_r2: float = MIN_R2,
) -> tuple[bool, str]:
    """Evalúa el quality gate y la comparación con el champion vigente.

    Returns:
        ``(promoted, reason)``. El motivo está pensado para logs e informes.
    """
    if test_rmse >= rmse_threshold:
        return (
            False,
            f"RMSE de test {test_rmse:.3f} no supera el umbral estricto "
            f"({rmse_threshold:.3f}); no se asigna el alias {CHAMPION_ALIAS}.",
        )
    if test_r2 <= min_r2:
        return (
            False,
            f"R² de test {test_r2:.3f} no supera el mínimo ({min_r2:.3f}); "
            f"no se asigna el alias {CHAMPION_ALIAS}.",
        )
    if current_champion_rmse is not None and test_rmse >= current_champion_rmse:
        return (
            False,
            f"RMSE de test {test_rmse:.3f} no mejora al champion vigente "
            f"({current_champion_rmse:.3f}); se conserva el modelo actual.",
        )
    return (
        True,
        f"RMSE de test {test_rmse:.3f} < {rmse_threshold:.3f} y "
        f"R² {test_r2:.3f} > {min_r2:.3f}; se promociona como {CHAMPION_ALIAS}.",
    )


def _current_champion_metadata(
    client: MlflowClient, model_name: str, alias: str
) -> tuple[str | None, float | None]:
    """Obtiene la versión y el RMSE del champion actual, si existe."""
    try:
        current = client.get_model_version_by_alias(model_name, alias)
    except MlflowException:
        return None, None

    previous_rmse: float | None = None
    try:
        run = client.get_run(current.run_id)
        previous_rmse = run.data.metrics.get("test_rmse")
    except MlflowException:
        logger.warning("No se pudieron leer las métricas del champion %s", current.version)
    return str(current.version), previous_rmse


def register_and_promote(
    result: TrainingResult,
    tracking_uri: str = TRACKING_URI,
    model_name: str = MODEL_NAME,
    alias: str = CHAMPION_ALIAS,
    rmse_threshold: float = RMSE_THRESHOLD,
    min_r2: float = MIN_R2,
) -> PromotionDecision:
    """Registra el mejor modelo y le asigna el alias de champion si pasa el gate."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    model_version = mlflow.register_model(result.model_uri, model_name)
    version = str(model_version.version)
    test_rmse = result.test_metrics["test_rmse"]
    test_r2 = result.test_metrics["test_r2"]

    description = (
        f"Candidato {result.model_family} con features {result.feature_set}. "
        f"Test RMSE={test_rmse:.3f}, MAE={result.test_metrics['test_mae']:.3f}, "
        f"R2={test_r2:.3f}. Run: {result.run_id}"
    )
    client.update_model_version(
        name=model_name,
        version=version,
        description=description,
    )
    version_tags: dict[str, Any] = {
        "model_family": result.model_family,
        "feature_set": result.feature_set,
        "test_rmse": f"{test_rmse:.6f}",
        "test_mae": f"{result.test_metrics['test_mae']:.6f}",
        "test_r2": f"{test_r2:.6f}",
        "run_id": result.run_id,
        "dataset_sha256": result.dataset_sha256,
    }
    for key, value in version_tags.items():
        client.set_model_version_tag(model_name, version, key, str(value))

    previous_version, previous_rmse = _current_champion_metadata(
        client, model_name, alias
    )
    promoted, reason = evaluate_promotion(
        test_rmse=test_rmse,
        test_r2=test_r2,
        current_champion_rmse=previous_rmse,
        rmse_threshold=rmse_threshold,
        min_r2=min_r2,
    )

    if promoted:
        client.set_registered_model_alias(model_name, alias, version)
        client.set_model_version_tag(model_name, version, "validation_status", "passed")
        logger.info("Modelo %s v%s promocionado como %s", model_name, version, alias)
    else:
        client.set_model_version_tag(model_name, version, "validation_status", "failed")
        logger.warning("Modelo %s v%s no promocionado: %s", model_name, version, reason)

    return PromotionDecision(
        promoted=promoted,
        reason=reason,
        version=version,
        alias=alias,
        previous_version=previous_version,
        previous_rmse=previous_rmse,
        test_rmse=test_rmse,
        test_r2=test_r2,
    )
