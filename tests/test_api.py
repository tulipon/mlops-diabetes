"""Tests de los endpoints y del manejo de errores de la API."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.model_loader import ChampionInfo, ChampionModel

VALID_PAYLOAD = {
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

INFO = ChampionInfo(
    model_name="diabetes-progression-model",
    version="7",
    alias="champion",
    run_id="run-123",
    feature_set="engineered",
    model_family="ridge",
    registered_at="2026-01-01T00:00:00+00:00",
    description="Modelo de prueba",
    metrics={"test_rmse": 52.5, "test_r2": 0.45},
)


class FakeModel:
    def __init__(self, value: float = 151.0, error: Exception | None = None) -> None:
        self.value = value
        self.error = error

    def predict(self, dataframe) -> np.ndarray:
        if self.error is not None:
            raise self.error
        return np.full(len(dataframe), self.value)


def make_champion(
    value: float = 151.0, error: Exception | None = None
) -> ChampionModel:
    return ChampionModel(FakeModel(value=value, error=error), INFO)


@pytest.fixture(autouse=True)
def reset_champion() -> Iterator[None]:
    app.state.champion = None
    yield
    app.state.champion = None


client = TestClient(app)


def test_health_is_ok_without_model() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": False}


def test_health_is_ok_with_model() -> None:
    app.state.champion = make_champion()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_model_info_returns_503_without_model() -> None:
    response = client.get("/model-info")
    assert response.status_code == 503
    assert "no está disponible" in response.json()["detail"]


def test_model_info_returns_metadata_with_model() -> None:
    app.state.champion = make_champion()
    response = client.get("/model-info")

    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == INFO.model_name
    assert body["model_version"] == INFO.version
    assert body["champion_alias"] == INFO.alias
    assert body["feature_set"] == INFO.feature_set
    assert body["metrics"]["test_rmse"] == INFO.metrics["test_rmse"]


def test_predict_returns_503_without_model() -> None:
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 503


def test_predict_returns_prediction_and_version() -> None:
    app.state.champion = make_champion(value=151.0)
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 151.0
    assert body["model_name"] == INFO.model_name
    assert body["model_version"] == INFO.version
    assert body["champion_alias"] == INFO.alias


def test_predict_rejects_missing_feature() -> None:
    app.state.champion = make_champion()
    payload = dict(VALID_PAYLOAD)
    payload.pop("bmi")

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_wrong_type() -> None:
    app.state.champion = make_champion()
    payload = {**VALID_PAYLOAD, "age": "no-es-un-numero"}

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_extra_field() -> None:
    app.state.champion = make_champion()
    payload = {**VALID_PAYLOAD, "campo_extra": 1.0}

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_returns_500_when_model_fails() -> None:
    app.state.champion = make_champion(error=RuntimeError("fallo simulado"))
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 500
    assert response.json()["detail"] == "No se pudo generar la predicción."
