"""Teste de integração ponta-a-ponta do orquestrador multimodal."""

from __future__ import annotations

from pathlib import Path

from multimodal_monitor.integration.orchestrator import MonitoringInput, PatientMonitor
from multimodal_monitor.schemas import Modality
from multimodal_monitor.synthetic import (
    SAMPLE_TRANSCRIPTS,
    generate_pose_frames,
    generate_prescriptions,
    generate_vitals,
)


def test_end_to_end_generates_alert(tmp_path: Path) -> None:
    # transcript de referência para o STT em modo mock
    audio = tmp_path / "consulta.wav"
    audio.with_suffix(".txt").write_text(SAMPLE_TRANSCRIPTS["critico"], encoding="utf-8")

    data = MonitoringInput(
        patient_id="PAC-TEST",
        vitals=generate_vitals(inject_anomalies=True),
        prescriptions=generate_prescriptions(inject_anomalies=True),
        pose_frames=generate_pose_frames(inject_anomalies=True),
        audio_path=audio,
    )
    monitor = PatientMonitor(dispatch_alerts=False)
    report = monitor.run(data)

    assert report.risk_score > 0.35
    assert report.alert is not None
    modalities = {r.modality for r in report.modality_results}
    assert Modality.VITALS in modalities
    assert Modality.PRESCRIPTION in modalities
    assert Modality.AUDIO in modalities
    assert Modality.VIDEO in modalities


def test_report_serialization() -> None:
    data = MonitoringInput(
        patient_id="PAC-2", vitals=generate_vitals(inject_anomalies=True)
    )
    report = PatientMonitor(dispatch_alerts=False).run(data)
    payload = report.as_dict()
    assert payload["patient_id"] == "PAC-2"
    assert "modalities" in payload
