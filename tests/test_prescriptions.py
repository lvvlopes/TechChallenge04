"""Testes do detector de anomalias em prescrições (RF03)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from multimodal_monitor.anomaly_detection.prescriptions import (
    PrescriptionAnomalyDetector,
    PrescriptionEvent,
)
from multimodal_monitor.schemas import Severity


def _t(offset_h: int = 0) -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=offset_h)


def test_detects_dose_above_max() -> None:
    events = [PrescriptionEvent(timestamp=_t(), drug="paracetamol", dose_mg=6000)]
    findings = PrescriptionAnomalyDetector().detect(events)
    assert any(f.metadata.get("rule") == "max_dose" for f in findings)
    assert any(f.severity in (Severity.HIGH, Severity.CRITICAL) for f in findings)


def test_detects_dose_jump() -> None:
    events = [
        PrescriptionEvent(timestamp=_t(0), drug="dipirona", dose_mg=1000),
        PrescriptionEvent(timestamp=_t(1), drug="dipirona", dose_mg=3000),
    ]
    findings = PrescriptionAnomalyDetector().detect(events)
    assert any(f.metadata.get("rule") == "dose_jump" for f in findings)


def test_detects_known_interaction() -> None:
    events = [
        PrescriptionEvent(timestamp=_t(0), drug="warfarina", dose_mg=5),
        PrescriptionEvent(timestamp=_t(1), drug="ibuprofeno", dose_mg=400),
    ]
    findings = PrescriptionAnomalyDetector().detect(events)
    assert any(f.metadata.get("rule") == "interaction" for f in findings)


def test_no_findings_on_safe_evolution() -> None:
    events = [
        PrescriptionEvent(timestamp=_t(0), drug="amoxicilina", dose_mg=500),
        PrescriptionEvent(timestamp=_t(1), drug="amoxicilina", dose_mg=500),
    ]
    assert PrescriptionAnomalyDetector().detect(events) == []
