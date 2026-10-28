"""Entrenamiento de candidatos y tracking de experimentos con MLflow."""

from __future__ import annotations

import logging
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import mlflow.data
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from mlops.config import (
    CV_FOLDS,
    EXPERIMENT_NAME,
    FEATURE_COLUMNS,
    MODEL_ARTIFACT_NAME,
    PROCESSED_DATA_PATH,
    PROJECT_ROOT,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    TRACKING_URI,
)
from mlops.data.extract import sha256_file
from mlops.features import FEATURE_SETS, build_model_pipeline

logger = logging.getLogger(__name__)


@dataclass
class CandidateResult:
    """Resultado de un candidato evaluado con validación cruzada."""

    model_family: str
    feature_set: str
    run_id: str
    cv_rmse: float
    cv_mae: float
    cv_r2: float
    cv_rmse_std: float
    best_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainingResult:
    """Resultado del entrenamiento y de la evaluación final en test."""

    run_id: str
    model_uri: str
    model_family: str
    feature_set: str
    best_params: dict[str, Any]
    cv_metrics: dict[str, float]
    test_metrics: dict[str, float]
    dataset_sha256: str
    train_rows: int
    test_rows: int
    candidates: list[CandidateResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["candidates"] = [candidate.to_dict() for candidate in self.candidates]
        return result


def candidate_models() -> list[tuple[str, Any, dict[str, list[Any]]]]:
    """Define los modelos candidatos y sus rejillas de hiperparámetros."""
    return [
        ("linear_regression", LinearRegression(), {}),
        ("ridge", Ridge(), {"model__alpha": [0.1, 1.0, 10.0]}),
        (
            "random_forest",
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 4],
            },
        ),
        (
            "gradient_boosting",
            GradientBoostingRegressor(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
            },
        ),
    ]


def _json_safe(value: Any) -> Any:
    """Convierte valores de hiperparámetros a tipos serializables en JSON."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def train_and_log(
    processed_path: str | Path = PROCESSED_DATA_PATH,
    tracking_uri: str = TRACKING_URI,
    experiment_name: str = EXPERIMENT_NAME,
    seed: int = RANDOM_STATE,
    test_size: float = TEST_SIZE,
    cv_folds: int = CV_FOLDS,
) -> TrainingResult:
    """Entrena todos los candidatos, registra cada run y evalúa el ganador.

    La selección del ganador se hace exclusivamente con RMSE de validación
    cruzada sobre train. El conjunto de test se evalúa una única vez al final
    para evitar leakage en la selección.
    """
    processed_path = Path(processed_path)
    dataframe = pd.read_csv(processed_path)

    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(dataframe.columns)
    if missing:
        raise ValueError(f"Faltan columnas en los datos procesados: {sorted(missing)}")

    X = dataframe[FEATURE_COLUMNS]
    y = dataframe[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    dataset_hash = sha256_file(processed_path)
    # MLflow 3 puede clasificar una ruta local con más de una fuente posible;
    # el warning es inocuo y se silencia para mantener los logs del pipeline claros.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The specified dataset source can be interpreted in multiple ways.*",
        )
        dataset = mlflow.data.from_pandas(
            dataframe,
            source=str(processed_path),
            name="diabetes_clean",
            targets=TARGET_COLUMN,
        )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    candidates: list[CandidateResult] = []
    fitted_pipelines: dict[str, Pipeline] = {}

    for feature_set in FEATURE_SETS:
        for model_family, estimator, param_grid in candidate_models():
            run_name = f"{model_family}-{feature_set}"
            logger.info("Entrenando candidato: %s", run_name)
            with mlflow.start_run(run_name=run_name) as run:
                run_id = run.info.run_id
                pipeline = build_model_pipeline(estimator, feature_set)

                mlflow.set_tags(
                    {
                        "stage": "candidate",
                        "model_family": model_family,
                        "feature_set": feature_set,
                        "dataset_sha256": dataset_hash,
                    }
                )
                mlflow.log_params(
                    {
                        "model_family": model_family,
                        "feature_set": feature_set,
                        "random_state": seed,
                        "test_size": test_size,
                        "cv_folds": cv_folds,
                    }
                )
                mlflow.log_input(dataset, context="training")

                search = GridSearchCV(
                    pipeline,
                    param_grid=param_grid or {},
                    scoring="neg_root_mean_squared_error",
                    cv=cv,
                    n_jobs=-1,
                    refit=True,
                )
                search.fit(X_train, y_train)

                cv_result = cross_validate(
                    search.best_estimator_,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring={
                        "rmse": "neg_root_mean_squared_error",
                        "mae": "neg_mean_absolute_error",
                        "r2": "r2",
                    },
                    n_jobs=-1,
                )
                cv_rmse_scores = -cv_result["test_rmse"]
                cv_mae_scores = -cv_result["test_mae"]
                cv_r2_scores = cv_result["test_r2"]

                cv_metrics = {
                    "cv_rmse": float(np.mean(cv_rmse_scores)),
                    "cv_mae": float(np.mean(cv_mae_scores)),
                    "cv_r2": float(np.mean(cv_r2_scores)),
                    "cv_rmse_std": float(np.std(cv_rmse_scores)),
                }
                best_params = {
                    key: _json_safe(value)
                    for key, value in search.best_params_.items()
                }

                mlflow.log_metrics(cv_metrics)
                mlflow.log_params(
                    {f"best__{key}": str(value) for key, value in best_params.items()}
                )
                mlflow.log_dict(
                    {
                        "best_params": best_params,
                        "cv_metrics": cv_metrics,
                    },
                    "cv_results.json",
                )

                candidates.append(
                    CandidateResult(
                        model_family=model_family,
                        feature_set=feature_set,
                        run_id=run_id,
                        cv_rmse=cv_metrics["cv_rmse"],
                        cv_mae=cv_metrics["cv_mae"],
                        cv_r2=cv_metrics["cv_r2"],
                        cv_rmse_std=cv_metrics["cv_rmse_std"],
                        best_params=best_params,
                    )
                )
                fitted_pipelines[run_id] = search.best_estimator_

    best = min(candidates, key=lambda candidate: (candidate.cv_rmse, candidate.cv_mae))
    best_pipeline = fitted_pipelines[best.run_id]
    predictions = best_pipeline.predict(X_test)

    test_metrics = {
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "test_mae": float(mean_absolute_error(y_test, predictions)),
        "test_r2": float(r2_score(y_test, predictions)),
    }

    logger.info(
        "Ganador por CV: %s (%s) con CV RMSE=%.3f. Test RMSE=%.3f.",
        best.model_family,
        best.feature_set,
        best.cv_rmse,
        test_metrics["test_rmse"],
    )

    with mlflow.start_run(run_id=best.run_id):
        mlflow.set_tags(
            {
                "stage": "best_model",
                "selected_by": "cv_rmse",
            }
        )
        mlflow.log_metrics(test_metrics)
        mlflow.log_params(
            {
                "selected_model_family": best.model_family,
                "selected_feature_set": best.feature_set,
            }
        )
        mlflow.log_dict(
            {
                "test_metrics": test_metrics,
                "candidates": [candidate.to_dict() for candidate in candidates],
            },
            "evaluation.json",
        )

        input_example = X_train.head(3)
        signature = infer_signature(
            X_train.head(20), best_pipeline.predict(X_train.head(20))
        )
        logged_model = mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            name=MODEL_ARTIFACT_NAME,
            signature=signature,
            input_example=input_example,
            code_paths=[str(PROJECT_ROOT / "mlops")],
        )
        model_uri = logged_model.model_uri

    return TrainingResult(
        run_id=best.run_id,
        model_uri=model_uri,
        model_family=best.model_family,
        feature_set=best.feature_set,
        best_params=best.best_params,
        cv_metrics={
            "cv_rmse": best.cv_rmse,
            "cv_mae": best.cv_mae,
            "cv_r2": best.cv_r2,
            "cv_rmse_std": best.cv_rmse_std,
        },
        test_metrics=test_metrics,
        dataset_sha256=dataset_hash,
        train_rows=int(len(X_train)),
        test_rows=int(len(X_test)),
        candidates=candidates,
    )
