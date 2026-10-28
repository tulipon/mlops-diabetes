"""Orquestador end-to-end del pipeline MLOps."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from mlops.config import (
    CHAMPION_ALIAS,
    CV_FOLDS,
    EXPERIMENT_NAME,
    MIN_R2,
    MODEL_NAME,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    RAW_DATA_PATH,
    REPORTS_DIR,
    RMSE_THRESHOLD,
    TEST_SIZE,
    TRACKING_URI,
)
from mlops.data.clean import clean_data, write_cleaning_report
from mlops.data.extract import extract_data, sha256_file
from mlops.data.validate import validate_data, write_validation_report
from mlops.training.registry import register_and_promote
from mlops.training.train import train_and_log

logger = logging.getLogger(__name__)


def run_pipeline(
    raw_path: str | Path = RAW_DATA_PATH,
    processed_path: str | Path = PROCESSED_DATA_PATH,
    reports_dir: str | Path = REPORTS_DIR,
    tracking_uri: str = TRACKING_URI,
    experiment_name: str = EXPERIMENT_NAME,
    model_name: str = MODEL_NAME,
    alias: str = CHAMPION_ALIAS,
    rmse_threshold: float = RMSE_THRESHOLD,
    min_r2: float = MIN_R2,
    seed: int = RANDOM_STATE,
    test_size: float = TEST_SIZE,
    cv_folds: int = CV_FOLDS,
) -> dict[str, Any]:
    """Ejecuta extracción, validación, limpieza, entrenamiento y registro."""
    raw_path = Path(raw_path)
    processed_path = Path(processed_path)
    reports_dir = Path(reports_dir)

    logger.info("1/5 Extrayendo datos")
    extract_data(raw_path)
    raw_dataframe = pd.read_csv(raw_path)

    logger.info("2/5 Validando datos crudos")
    raw_report = validate_data(raw_dataframe, allow_nulls=True)
    write_validation_report(raw_report, reports_dir / "validation_report.json")

    logger.info("3/5 Limpiando datos")
    cleaned_dataframe, cleaning_stats = clean_data(raw_dataframe)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_dataframe.to_csv(processed_path, index=False)
    write_cleaning_report(cleaning_stats, reports_dir / "cleaning_report.json")

    cleaned_report = validate_data(cleaned_dataframe)
    write_validation_report(
        cleaned_report, reports_dir / "validation_report_clean.json"
    )

    logger.info("4/5 Entrenando y registrando runs en MLflow")
    training_result = train_and_log(
        processed_path=processed_path,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        seed=seed,
        test_size=test_size,
        cv_folds=cv_folds,
    )

    logger.info("5/5 Registrando modelo y evaluando el quality gate")
    decision = register_and_promote(
        result=training_result,
        tracking_uri=tracking_uri,
        model_name=model_name,
        alias=alias,
        rmse_threshold=rmse_threshold,
        min_r2=min_r2,
    )

    summary = {
        "dataset": {
            "raw_path": str(raw_path),
            "processed_path": str(processed_path),
            "raw_sha256": sha256_file(raw_path),
            "processed_sha256": training_result.dataset_sha256,
            "train_rows": training_result.train_rows,
            "test_rows": training_result.test_rows,
            "cleaning": cleaning_stats.to_dict(),
        },
        "experiment": {
            "tracking_uri": tracking_uri,
            "experiment_name": experiment_name,
            "model_name": model_name,
            "run_id": training_result.run_id,
            "model_uri": training_result.model_uri,
        },
        "training": training_result.to_dict(),
        "champion": decision.to_dict(),
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "pipeline_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta el pipeline MLOps completo de diabetes."
    )
    parser.add_argument("--tracking-uri", default=TRACKING_URI)
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--alias", default=CHAMPION_ALIAS)
    parser.add_argument("--rmse-threshold", type=float, default=RMSE_THRESHOLD)
    parser.add_argument("--min-r2", type=float, default=MIN_R2)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--cv-folds", type=int, default=CV_FOLDS)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()
    summary = run_pipeline(
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        model_name=args.model_name,
        alias=args.alias,
        rmse_threshold=args.rmse_threshold,
        min_r2=args.min_r2,
        seed=args.seed,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
    )

    training = summary["training"]
    champion = summary["champion"]
    print("\nPipeline completado")
    print(f"  Run ganador: {training['run_id']}")
    print(
        "  Modelo: "
        f"{training['model_family']} ({training['feature_set']})"
    )
    print(
        "  CV RMSE: "
        f"{training['cv_metrics']['cv_rmse']:.3f} | "
        "Test RMSE: "
        f"{training['test_metrics']['test_rmse']:.3f} | "
        "Test R2: "
        f"{training['test_metrics']['test_r2']:.3f}"
    )
    print(
        f"  Champion: v{champion['version']} "
        f"{'promocionado' if champion['promoted'] else 'NO promocionado'}"
    )
    print(f"  Motivo: {champion['reason']}")
    print(f"  Informe: {REPORTS_DIR / 'pipeline_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
