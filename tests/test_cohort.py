"""Testes da coorte de pacientes e dos endpoints de análise por ID."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
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


def _build_patient(root: Path, pid: str, scenario: str, seed: int) -> None:
    """Cria em disco os arquivos de um paciente (mesma estrutura do gerador)."""
    critical = scenario == "critico"
    pdir = root / pid
    pdir.mkdir(parents=True, exist_ok=True)
    generate_vitals(seed=seed, inject_anomalies=critical).to_csv(pdir / "vitals.csv", index=False)
    presc = generate_prescriptions(seed=seed, inject_anomalies=critical)
    pd.DataFrame([p.model_dump() for p in presc]).to_csv(pdir / "prescriptions.csv", index=False)
    poses = generate_pose_frames(seed=seed, inject_anomalies=critical)
    pd.DataFrame([p.__dict__ for p in poses]).to_csv(pdir / "pose_frames.csv", index=False)
    pool = CRITICAL_TRANSCRIPTS if critical else STABLE_TRANSCRIPTS
    (pdir / "consulta.txt").write_text(pool[0], encoding="utf-8")


@pytest.fixture
def cohort_root(tmp_path: Path) -> Path:
    root = tmp_path / "patients"
    root.mkdir()
    manifest = {
        "count": 2,
        "patients": [
            {
                "id": "PAC-001", "name": "Ana Ribeiro", "age": 60,
                "bed": "UTI · Leito 07", "scenario": "critico",
                "chief_complaint": "Dispneia", "video": None,
            },
            {
                "id": "PAC-002", "name": "Bruno Carvalho", "age": 45,
                "bed": "Enfermaria · Leito 03", "scenario": "estavel",
                "chief_complaint": "Rotina", "video": None,
            },
        ],
    }
    (root / "cohort.json").write_text(json.dumps(manifest), encoding="utf-8")
    _build_patient(root, "PAC-001", "critico", seed=101)
    _build_patient(root, "PAC-002", "estavel", seed=102)
    return root


def test_cohort_lists_patients(cohort_root: Path) -> None:
    cohort = PatientCohort(cohort_root)
    patients = cohort.list_patients()
    assert len(patients) == 2
    assert {p.id for p in patients} == {"PAC-001", "PAC-002"}


def test_cohort_load_input_binds_all_modalities(cohort_root: Path) -> None:
    data = PatientCohort(cohort_root).load_input("PAC-001")
    assert data.patient_id == "PAC-001"
    assert data.vitals is not None and not data.vitals.empty
    assert data.prescriptions  # crítico tem anomalias de prescrição
    assert data.pose_frames  # pose pré-computada carregada
    assert data.audio_path is not None  # aponta para .wav (com .txt irmão)


def test_cohort_unknown_patient_raises(cohort_root: Path) -> None:
    with pytest.raises(KeyError):
        PatientCohort(cohort_root).load_input("PAC-999")


def test_critical_patient_scores_higher_than_stable(cohort_root: Path) -> None:
    from multimodal_monitor.integration.orchestrator import PatientMonitor

    cohort = PatientCohort(cohort_root)
    monitor = PatientMonitor(dispatch_alerts=False)
    crit = monitor.run(cohort.load_input("PAC-001"))
    stable = monitor.run(cohort.load_input("PAC-002"))
    assert crit.risk_score > stable.risk_score
    assert crit.alert is not None
    assert stable.alert is None  # estável não dispara alerta


def test_api_list_and_analyze(monkeypatch, cohort_root: Path) -> None:
    # aponta a API para a coorte de teste
    import multimodal_monitor.api.main as main

    monkeypatch.setattr(main, "_get_cohort", lambda: PatientCohort(cohort_root))
    with TestClient(main.app) as c:
        r = c.get("/api/patients")
        assert r.status_code == 200
        assert r.json()["available"] is True
        assert len(r.json()["patients"]) == 2

        r2 = c.post("/api/patients/PAC-001/analyze", json={"use_real_video": False})
        assert r2.status_code == 200
        body = r2.json()
        assert body["patient"]["id"] == "PAC-001"
        assert body["risk_score"] > 0.35
        assert body["alert"] is not None

        r3 = c.post("/api/patients/PAC-999/analyze", json={})
        assert r3.status_code == 404
