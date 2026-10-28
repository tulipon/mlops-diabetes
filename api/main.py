"""API FastAPI para servir el modelo champion de progresión de diabetes."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated, AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from api.model_loader import ChampionModel, load_champion
from api.schemas import (
    DiabetesFeatures,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Carga el champion al arrancar; la API sigue viva si no está disponible."""
    try:
        champion = load_champion()
        app.state.champion = champion
        app.state.load_error = None
        logger.info(
            "Champion cargado: %s v%s",
            champion.info.model_name,
            champion.info.version,
        )
    except Exception as exc:  # noqa: BLE001 - la API debe arrancar en modo degradado
        app.state.champion = None
        app.state.load_error = str(exc)
        logger.exception("No se pudo cargar el modelo champion: %s", exc)
    yield


app = FastAPI(
    title="Diabetes Progression API",
    description=(
        "API de inferencia para el modelo champion de progresión de diabetes. "
        "Expone el modelo registrado en MLflow con el alias `champion`."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def get_champion(request: Request) -> ChampionModel | None:
    """Dependencia que devuelve el champion cargado o ``None``."""
    return getattr(request.app.state, "champion", None)


ChampionDependency = Annotated[ChampionModel | None, Depends(get_champion)]


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {
        "service": "Diabetes Progression API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
def health(champion: ChampionDependency) -> HealthResponse:
    """Comprueba que el servicio está vivo y si el modelo está cargado."""
    return HealthResponse(status="ok", model_loaded=champion is not None)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info(champion: ChampionDependency) -> ModelInfoResponse:
    """Devuelve la versión y metadatos del modelo que se está sirviendo."""
    if champion is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo champion no está disponible.",
        )

    info = champion.info
    return ModelInfoResponse(
        model_name=info.model_name,
        model_version=info.version,
        champion_alias=info.alias,
        run_id=info.run_id,
        feature_set=info.feature_set,
        model_family=info.model_family,
        registered_at=info.registered_at,
        description=info.description,
        metrics=info.metrics,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(
    payload: DiabetesFeatures,
    champion: ChampionDependency,
) -> PredictionResponse:
    """Genera una predicción para una fila de features estandarizadas."""
    if champion is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo champion no está disponible.",
        )

    try:
        prediction = champion.predict(payload.model_dump())
    except Exception as exc:  # noqa: BLE001 - se traduce a un 500 controlado
        logger.exception("Error durante la predicción")
        raise HTTPException(
            status_code=500,
            detail="No se pudo generar la predicción.",
        ) from exc

    return PredictionResponse(
        prediction=prediction,
        model_name=champion.info.model_name,
        model_version=champion.info.version,
        champion_alias=champion.info.alias,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Evita filtrar trazas internas en errores no controlados."""
    logger.exception("Error no controlado en %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Error interno del servidor."},
    )
