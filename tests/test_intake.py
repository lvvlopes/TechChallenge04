"""Testes do endpoint POST /intake (captura clínica multipart)."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from multimodal_monitor.api.main import app


def _client() -> TestClient:
    return TestClient(app)


def test_intake_empty_returns_ok_and_no_alert() -> None:
    """Sem nenhum dado clínico, o endpoint responde 200 e não gera alerta."""
    with _client() as c:
        r = c.post("/intake", data={"patient_id": "PAC-EMPTY"})
    assert r.status_code == 200
    body = r.json()
    assert body["patient_id"] == "PAC-EMPTY"
    assert body["risk_score"] == 0.0
    assert body["alert"] is None
    assert body["inputs_received"]["symptoms_text"] is False
    assert body["inputs_received"]["vitals_readings"] == 0


def test_intake_symptoms_only_detects_critical_terms() -> None:
    """Sintomas em texto livre viram achados na modalidade AUDIO."""
    payload = {"symptoms": "Doutor, estou com muita falta de ar e dor no peito."}
    with _client() as c:
        r = c.post(
            "/intake",
            data={"patient_id": "PAC-SX", "data": json.dumps(payload)},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["inputs_received"]["symptoms_text"] is True
    assert "audio" in body["modalities"]
    descriptions = [
        f["description"] for f in body["modalities"]["audio"]["findings"]
    ]
    assert any("dor no peito" in d for d in descriptions)
    assert any("falta de ar" in d for d in descriptions)


def test_intake_vitals_and_symptoms_triggers_alert() -> None:
    """Combinação clínica realista deve gerar alerta de alta severidade."""
    payload = {
        "symptoms": "Estou com muita falta de ar e dor no peito, muito mal.",
        "readings": [
            {
                "timestamp": "2026-07-11T10:00:00Z",
                "spo2": 82.0,
                "heart_rate": 132.0,
                "systolic_bp": 185.0,
            }
        ],
        "prescriptions": [
            {
                "timestamp": "2026-07-11T09:00:00Z",
                "drug": "warfarina",
                "dose_mg": 5,
                "action": "prescribe",
            },
            {
                "timestamp": "2026-07-11T10:30:00Z",
                "drug": "ibuprofeno",
                "dose_mg": 600,
                "action": "prescribe",
            },
        ],
    }
    with _client() as c:
        r = c.post(
            "/intake",
            data={"patient_id": "PAC-ALL", "data": json.dumps(payload)},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["alert"] is not None
    assert body["alert"]["severity"] in ("high", "critical")
    assert body["risk_score"] > 0.35
    # todas as três modalidades presentes
    mods = set(body["modalities"].keys())
    assert "audio" in mods
    assert "vitals" in mods
    assert "prescription" in mods


def test_intake_invalid_json_returns_400() -> None:
    with _client() as c:
        r = c.post(
            "/intake",
            data={"patient_id": "PAC-BAD", "data": "isto nao eh json"},
        )
    assert r.status_code == 400
    assert "JSON" in r.json()["detail"]


def test_intake_audio_upload_is_accepted() -> None:
    """Upload de áudio: o endpoint aceita o arquivo sem erro.

    Sem transcrição real (SDK/credenciais ausentes) e sem `.txt` irmão
    no diretório temporário, o STT em modo mock retorna texto vazio —
    mas o endpoint continua respondendo 200.
    """
    fake_wav = b"RIFF\x00\x00\x00\x00WAVEfmt "  # bytes bogus só para o upload
    with _client() as c:
        r = c.post(
            "/intake",
            data={"patient_id": "PAC-AUDIO"},
            files={"audio": ("consulta.wav", fake_wav, "audio/wav")},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["inputs_received"]["audio_file"] == "consulta.wav"


def test_intake_rejects_non_object_data() -> None:
    with _client() as c:
        r = c.post(
            "/intake",
            data={"patient_id": "PAC-ARR", "data": "[1,2,3]"},
        )
    assert r.status_code == 400
