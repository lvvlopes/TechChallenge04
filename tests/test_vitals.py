"""Testes do detector de anomalias em sinais vitais (RF03)."""

from __future__ import annotations

from multimodal_monitor.anomaly_detection.vitals import VitalsAnomalyDetector
from multimodal_monitor.schemas import Severity
from multimodal_monitor.synthetic import generate_vitals


def test_no_findings_on_clean_series() -> None:
    df = generate_vitals(n=200, inject_anomalies=False, seed=1)
    findings = VitalsAnomalyDetector().detect(df)
    # série limpa pode gerar poucos outliers estatísticos, mas nenhum crítico
    assert all(f.severity is not Severity.CRITICAL for f in findings)


def test_detects_injected_desaturation() -> None:
    df = generate_vitals(n=240, inject_anomalies=True, seed=42)
    findings = VitalsAnomalyDetector().detect(df)
    spo2_findings = [f for f in findings if f.metadata.get("signal") == "spo2"]
    assert spo2_findings, "esperava detectar dessaturação de SpO2"
    assert any(f.severity in (Severity.MEDIUM, Severity.CRITICAL) for f in spo2_findings)


def test_empty_dataframe_returns_empty() -> None:
    import pandas as pd

    assert VitalsAnomalyDetector().detect(pd.DataFrame()) == []


def test_critical_when_spo2_below_threshold() -> None:
    from datetime import datetime, timezone

    import pandas as pd

    df = pd.DataFrame(
        {
            "timestamp": [datetime.now(timezone.utc)],
            "spo2": [82.0],  # abaixo do limite crítico (88)
        }
    )
    findings = VitalsAnomalyDetector().detect(df)
    assert any(f.severity is Severity.CRITICAL for f in findings)
