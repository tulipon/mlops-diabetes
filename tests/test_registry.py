"""Tests del quality gate y de la lógica de promoción."""

from __future__ import annotations

from mlops.training.registry import evaluate_promotion


def test_promotion_passes_with_good_metrics() -> None:
    promoted, reason = evaluate_promotion(test_rmse=53.0, test_r2=0.45)
    assert promoted is True
    assert "se promociona" in reason


def test_promotion_fails_when_rmse_is_not_below_strict_threshold() -> None:
    promoted, reason = evaluate_promotion(test_rmse=54.0, test_r2=0.50)
    assert promoted is False
    assert "umbral estricto" in reason


def test_promotion_fails_when_r2_is_too_low() -> None:
    promoted, reason = evaluate_promotion(test_rmse=50.0, test_r2=0.40)
    assert promoted is False
    assert "R²" in reason


def test_promotion_fails_when_it_does_not_beat_current_champion() -> None:
    promoted, reason = evaluate_promotion(
        test_rmse=53.0,
        test_r2=0.45,
        current_champion_rmse=52.0,
    )
    assert promoted is False
    assert "no mejora" in reason


def test_promotion_passes_when_it_beats_current_champion() -> None:
    promoted, reason = evaluate_promotion(
        test_rmse=51.0,
        test_r2=0.45,
        current_champion_rmse=52.0,
    )
    assert promoted is True
    assert "se promociona" in reason
