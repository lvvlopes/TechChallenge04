"""Testes do detector de anomalias de movimentação (RF03)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from multimodal_monitor.anomaly_detection.movement import MovementAnomalyDetector


def _timestamps(n: int) -> list[datetime]:
    t0 = datetime.now(timezone.utc)
    return [t0 + timedelta(seconds=i) for i in range(n)]


def test_detects_prolonged_immobility() -> None:
    n = 40
    movement = [0.3] * n
    movement[5:25] = [0.01] * 20  # 20 intervalos imóveis
    findings = MovementAnomalyDetector().detect(_timestamps(n), movement)
    assert any(f.metadata.get("rule") == "immobility" for f in findings)


def test_detects_movement_spike() -> None:
    movement = [0.2] * 30
    movement[15] = 1.0
    findings = MovementAnomalyDetector().detect(_timestamps(30), movement)
    assert any(f.metadata.get("rule") == "spike" for f in findings)


def test_short_series_returns_empty() -> None:
    assert MovementAnomalyDetector().detect(_timestamps(2), [0.1, 0.2]) == []
