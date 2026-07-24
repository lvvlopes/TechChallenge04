"""Testes da coorte de pacientes (arquivo único) e dos endpoints por ID."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from multimodal_monitor.cohort import PatientCohort
from multimodal_monitor.synthetic import (
    CRITICAL_TRANSCRIPTS,
    STABLE_TRANSCRIPTS,
    generate_pose_frames,
    generate_prescriptions,
    generate_vitals,
)

ROUND = {
    "heart_rate": 1, "systolic_bp": 1, "diastolic_bp": 1,
    "spo2": 1, "respiratory_rate": 1, "temperature": 2,
}


def _build_patient(pid: str, name: str, scenario: str, seed: int) -> dict:
    """Monta o dicionário de um paciente no mesmo formato do gerador."""
    critical = scenario == "critico"
    vdf = generate_vitals(seed=seed, inject_anomalies=critical)
    vitals = {"timestamp": [ts.isoformat() for ts in vdf["timestamp"]]}
    for col, nd in ROUND.items():
        vitals[col] = [round(float(v), nd) for v in vdf[col]]

    poses = generate_pose_frames(seed=seed, inject_anomalies=critical)
    pool = CRITICAL_TRANSCRIPTS if critical else STABLE_TRANSCRIPTS

    return {
        "id": pid,
        "name": name,
        "age": 60,
        "bed": "UTI · Leito 07",
        "scenario": scenario,
        "chief_complaint": "Teste",
        "transcript": pool[0],
        "vitals": vitals,
        "prescriptions": [
            {
                "timestamp": p.timestamp.isoformat(),
                "drug": p.drug,
                "dose_mg": p.dose_mg,
                "action": p.action,
            }
            for p in generate_prescriptions(seed=seed, inject_anomalies=critical)
        ],
        "pose_frames": {
            "frame_index": [p.frame_index for p in poses],
            "timestamp_s": [round(p.timestamp_s, 3) for p in poses],
            "movement_index": [round(p.movement_index, 4) for p in poses],
            "trunk_angle": [
                None if p.trunk_angle is None else round(p.trunk_angle, 2) for p in poses
            ],
        },
    }


@pytest.fixture
def cohort_file(tmp_path: Path) -> Path:
    """Cria um data/patients.json de teste com 2 pacientes."""
    (tmp_path / "patients_media").mkdir()
    manifest = {
        "count": 2,
        "media_dir": "patients_media",
        "patients": [
            _build_patient("PAC-001", "Ana Ribeiro", "critico", seed=101),
            _build_patient("PAC-002", "Bruno Carvalho", "estavel", seed=102),
        ],
    }
    path = tmp_path / "patients.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return path


def test_cohort_lists_patients(cohort_file: Path) -> None:
    patients = PatientCohort(cohort_file).list_patients()
    assert len(patients) == 2
    assert {p.id for p in patients} == {"PAC-001", "PAC-002"}
    assert patients[0].name == "Ana Ribeiro"


def test_cohort_load_input_binds_all_modalities(cohort_file: Path) -> None:
    data = PatientCohort(cohort_file).load_input("PAC-001")
    assert data.patient_id == "PAC-001"
    assert data.vitals is not None and not data.vitals.empty
    assert list(data.vitals.columns)  # colunas reconstruídas do formato colunar
    assert data.prescriptions           # crítico tem anomalias de prescrição
    assert data.pose_frames             # pose carregada do JSON
    assert data.transcript              # transcrição vem do próprio dataset


def test_cohort_unknown_patient_raises(cohort_file: Path) -> None:
    with pytest.raises(KeyError):
        PatientCohort(cohort_file).load_input("PAC-999")


def test_missing_cohort_file_is_unavailable(tmp_path: Path) -> None:
    cohort = PatientCohort(tmp_path / "nao_existe.json")
    assert cohort.available is False
    assert cohort.list_patients() == []


def test_critical_patient_scores_higher_than_stable(cohort_file: Path) -> None:
    from multimodal_monitor.integration.orchestrator import PatientMonitor

    cohort = PatientCohort(cohort_file)
    monitor = PatientMonitor(dispatch_alerts=False)
    crit = monitor.run(cohort.load_input("PAC-001"))
    stable = monitor.run(cohort.load_input("PAC-002"))
    assert crit.risk_score > stable.risk_score
    assert crit.alert is not None
    assert stable.alert is None  # estável não dispara alerta


def test_transcript_feeds_audio_modality(cohort_file: Path) -> None:
    """Sem arquivo de áudio, a transcrição do dataset alimenta a modalidade."""
    from multimodal_monitor.integration.orchestrator import PatientMonitor
    from multimodal_monitor.schemas import Modality

    report = PatientMonitor(dispatch_alerts=False).run(
        PatientCohort(cohort_file).load_input("PAC-001")
    )
    audio = next(
        (r for r in report.modality_results if r.modality is Modality.AUDIO), None
    )
    assert audio is not None
    assert audio.findings, "termos críticos deveriam vir da transcrição do dataset"


def test_api_list_and_analyze(monkeypatch, cohort_file: Path) -> None:
    import multimodal_monitor.api.main as main

    monkeypatch.setattr(main, "_get_cohort", lambda: PatientCohort(cohort_file))
    with TestClient(main.app) as c:
        r = c.get("/api/patients")
        assert r.status_code == 200
        assert r.json()["available"] is True
        assert len(r.json()["patients"]) == 2

        r2 = c.post("/api/patients/PAC-001/analyze", json={})
        assert r2.status_code == 200
        body = r2.json()
        assert body["patient"]["id"] == "PAC-001"
        assert body["risk_score"] > 0.35
        assert body["alert"] is not None
        assert body["has_video_file"] is False  # sem .mp4 na mídia de teste

        assert c.post("/api/patients/PAC-999/analyze", json={}).status_code == 404
