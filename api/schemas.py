"""Esquemas Pydantic de entrada y salida de la API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DiabetesFeatures(BaseModel):
    """Features estandarizadas del dataset de diabetes de scikit-learn.

    Los valores no están en unidades clínicas originales: ``load_diabetes``
    devuelve las variables ya centradas y escaladas. El README incluye un
    ejemplo válido.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    age: float = Field(..., description="Edad estandarizada")
    sex: float = Field(..., description="Sexo estandarizado")
    bmi: float = Field(..., description="Índice de masa corporal estandarizado")
    bp: float = Field(..., description="Presión arterial media estandarizada")
    s1: float = Field(..., description="Colesterol total sérico estandarizado")
    s2: float = Field(..., description="LDL estandarizado")
    s3: float = Field(..., description="HDL estandarizado")
    s4: float = Field(..., description="Ratio colesterol/HDL estandarizado")
    s5: float = Field(..., description="Logaritmo de triglicéridos estandarizado")
    s6: float = Field(..., description="Glucosa sérica estandarizada")


class PredictionResponse(BaseModel):
    prediction: float
    model_name: str
    model_version: str
    champion_alias: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    champion_alias: str
    run_id: str | None = None
    feature_set: str | None = None
    model_family: str | None = None
    registered_at: str | None = None
    description: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
