"""API REST para monitoramento multimodal near real-time.

Endpoints
---------
- ``GET  /health``            — status e capacidades disponíveis (Azure/visão).
- ``POST /monitor/vitals``    — analisa um lote de leituras de sinais vitais.
- ``POST /monitor/prescriptions`` — analisa uma evolução de prescrições.
- ``POST /monitor``           — ciclo multimodal (vitais + prescrições + texto).
- ``POST /intake``            — captura clínica multipart: sintomas digitados +
                                vitais + prescrições + upload de áudio/vídeo.

Execução: ``uvicorn multimodal_monitor.api.main:app --reload``
"""

from __future__ import annotations

import json
import logging
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..anomaly_detection.prescriptions import PrescriptionEvent
from ..cohort import PatientCohort
from ..config import get_settings
from ..integration.orchestrator import MonitoringInput, PatientMonitor
from ..schemas import Finding, Modality, ModalityResult, VitalSignReading
from ..synthetic import (
    SAMPLE_TRANSCRIPTS,
    generate_pose_frames,
    generate_prescriptions,
    generate_vitals,
)

_STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Monitoramento Multimodal de Pacientes",
    description=(
        "IA para monitoramento contínuo de pacientes (áudio, vídeo, texto) — FIAP TC Fase 4."
    ),
    version="0.1.0",
)

if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def dashboard_root() -> FileResponse:
    """Serve o dashboard interativo na raiz."""
    return FileResponse(_STATIC_DIR / "dashboard.html")


@app.get("/dashboard", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(_STATIC_DIR / "dashboard.html")


@app.get("/intake", include_in_schema=False)
def intake_page() -> FileResponse:
    """Serve a tela de captura clínica (formulário multimodal com gravação)."""
    return FileResponse(_STATIC_DIR / "intake.html")


@app.get("/patients", include_in_schema=False)
def patients_page() -> FileResponse:
    """Serve a tela de seleção e análise de pacientes da coorte."""
    return FileResponse(_STATIC_DIR / "patients.html")


# --------------------------------------------------------------------------- #
# Modelos de requisição
# --------------------------------------------------------------------------- #
class VitalsRequest(BaseModel):
    patient_id: str = Field(examples=["PAC-001"])
    readings: list[VitalSignReading]


class PrescriptionRequest(BaseModel):
    patient_id: str = Field(examples=["PAC-001"])
    events: list[PrescriptionEvent]


class MonitorRequest(BaseModel):
    patient_id: str = Field(examples=["PAC-001"])
    readings: list[VitalSignReading] = Field(default_factory=list)
    prescriptions: list[PrescriptionEvent] = Field(default_factory=list)
    transcript: str | None = Field(
        default=None,
        description="Texto já transcrito da consulta (alternativa ao upload de áudio).",
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _readings_to_df(readings: list[VitalSignReading]) -> pd.DataFrame:
    if not readings:
        return pd.DataFrame()
    return pd.DataFrame([r.model_dump() for r in readings])


def _get_monitor() -> PatientMonitor:
    # alertas não são despachados via API por padrão; retornados no corpo.
    return PatientMonitor(dispatch_alerts=False)


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict:
    settings = get_settings()
    monitor = _get_monitor()
    return {
        "status": "ok",
        "version": app.version,
        "time": datetime.now().isoformat(),
        "capabilities": {
            "azure_speech": settings.azure_speech_enabled,
            "azure_language": settings.azure_language_enabled,
            "vision_pose": monitor.video_pipeline.pose_analyzer.available,
            "object_detection": bool(
                monitor.video_pipeline.object_detector
                and monitor.video_pipeline.object_detector.available
            ),
        },
    }


@app.post("/monitor/vitals")
def monitor_vitals(req: VitalsRequest) -> dict:
    monitor = _get_monitor()
    report = monitor.run(
        MonitoringInput(patient_id=req.patient_id, vitals=_readings_to_df(req.readings))
    )
    return report.as_dict()


@app.post("/monitor/prescriptions")
def monitor_prescriptions(req: PrescriptionRequest) -> dict:
    monitor = _get_monitor()
    report = monitor.run(
        MonitoringInput(patient_id=req.patient_id, prescriptions=req.events)
    )
    return report.as_dict()


@app.post("/monitor")
def monitor(req: MonitorRequest) -> dict:
    """Ciclo multimodal combinando sinais vitais, prescrições e texto de consulta."""
    monitor = _get_monitor()
    results = []

    # vitais + prescrições via orquestrador
    report = monitor.run(
        MonitoringInput(
            patient_id=req.patient_id,
            vitals=_readings_to_df(req.readings),
            prescriptions=req.prescriptions,
        )
    )
    results = report.modality_results

    # texto de consulta (quando fornecido diretamente) via análise de texto
    if req.transcript:
        insights = monitor.audio_pipeline.text_analytics.analyze(req.transcript)

        findings = [
            Finding(
                modality=Modality.AUDIO,
                severity=sev,
                description=f"Termo crítico mencionado: '{term}'",
                score=0.6,
                metadata={"term": term},
            )
            for term, sev in insights.critical_terms
        ]
        text_result = ModalityResult(
            modality=Modality.AUDIO,
            findings=findings,
            summary=(
                f"Sentimento: {insights.sentiment} ({insights.sentiment_score:+.2f}) "
                f"[{insights.source}]"
            ),
            metadata={
                "sentiment": insights.sentiment,
                "sentiment_score": insights.sentiment_score,
                "sentiment_source": insights.source,
            },
        )
        results = results + [text_result]

    # refunde com todas as modalidades (incluindo texto)
    risk_score, alert = monitor.fusion.fuse(req.patient_id, results)
    return {
        "patient_id": req.patient_id,
        "risk_score": risk_score,
        "alert": alert.model_dump(mode="json") if alert else None,
        "modalities": {
            r.modality.value: {
                "risk_score": r.risk_score,
                "summary": r.summary,
                "findings": [f.model_dump(mode="json") for f in r.findings],
                "metadata": r.metadata,
            }
            for r in results
        },
    }


class DemoRequest(BaseModel):
    """Parâmetros do ciclo de demonstração com dados sintéticos."""

    scenario: str = Field(default="critical", description="critical | stable")
    patient_id: str = Field(default="PAC-001")
    seed: int = 42


@app.post("/demo/run")
def demo_run(req: DemoRequest) -> dict:
    """Executa um ciclo multimodal com dados sintéticos.

    Usado pelo dashboard para gerar demonstrações reprodutíveis. Retorna, além
    do relatório multimodal, uma amostra dos sinais vitais e da transcrição
    para renderização no cliente.
    """
    inject = req.scenario.lower() == "critical"
    monitor = _get_monitor()

    vitals_df = generate_vitals(seed=req.seed, inject_anomalies=inject)
    prescriptions = generate_prescriptions(seed=req.seed, inject_anomalies=inject)
    pose_frames = generate_pose_frames(seed=req.seed, inject_anomalies=inject)
    transcript = SAMPLE_TRANSCRIPTS["critico" if inject else "estavel"]

    report = monitor.run(
        MonitoringInput(
            patient_id=req.patient_id,
            vitals=vitals_df,
            prescriptions=prescriptions,
            pose_frames=pose_frames,
        )
    )

    # análise textual (equivale ao STT em modo mock, sem I/O de arquivo)
    insights = monitor.audio_pipeline.text_analytics.analyze(transcript)

    audio_result = ModalityResult(
        modality=Modality.AUDIO,
        findings=[
            Finding(
                modality=Modality.AUDIO,
                severity=sev,
                description=f"Termo crítico mencionado: '{term}'",
                score=0.6,
                metadata={"term": term},
            )
            for term, sev in insights.critical_terms
        ],
        summary=(
            f"Sentimento: {insights.sentiment} "
            f"({insights.sentiment_score:+.2f}); "
            f"termos críticos: {len(insights.critical_terms)}"
        ),
    )
    all_results = list(report.modality_results) + [audio_result]
    risk_score, alert = monitor.fusion.fuse(req.patient_id, all_results)

    return {
        "patient_id": req.patient_id,
        "scenario": req.scenario,
        "generated_at": datetime.now().isoformat(),
        "risk_score": risk_score,
        "alert": alert.model_dump(mode="json") if alert else None,
        "modalities": {
            r.modality.value: {
                "risk_score": r.risk_score,
                "summary": r.summary,
                "findings": [f.model_dump(mode="json") for f in r.findings],
            }
            for r in all_results
        },
        "vitals_series": {
            "timestamps": [ts.isoformat() for ts in vitals_df["timestamp"]],
            "heart_rate": vitals_df["heart_rate"].tolist(),
            "systolic_bp": vitals_df["systolic_bp"].tolist(),
            "diastolic_bp": vitals_df["diastolic_bp"].tolist(),
            "spo2": vitals_df["spo2"].tolist(),
            "respiratory_rate": vitals_df["respiratory_rate"].tolist(),
            "temperature": vitals_df["temperature"].tolist(),
        },
        "audio": {
            "transcript": transcript,
            "sentiment": insights.sentiment,
            "sentiment_score": insights.sentiment_score,
            "critical_terms": [
                {"term": t, "severity": s.value} for t, s in insights.critical_terms
            ],
        },
    }


# --------------------------------------------------------------------------- #
# POST /intake — captura clínica multipart (upload de áudio/vídeo + JSON)
# --------------------------------------------------------------------------- #
async def _save_upload(upload: UploadFile, dst_dir: Path) -> Path:
    """Persiste um `UploadFile` em disco e devolve o caminho gerado.

    Sanitiza o nome (usa apenas o basename) para evitar path traversal.
    """
    filename = Path(upload.filename or "upload.bin").name
    dst = dst_dir / filename
    content = await upload.read()
    dst.write_bytes(content)
    return dst


def _merge_or_append_audio(
    results: list[ModalityResult],
    typed_findings: list[Finding],
    typed_summary: str,
) -> list[ModalityResult]:
    """Se já existe um `ModalityResult` de AUDIO nos resultados, mescla os
    achados dos sintomas digitados nele; caso contrário, cria um novo.

    Mantém uma única entrada de AUDIO na fusão (evita dupla contagem).
    """
    existing = next((r for r in results if r.modality == Modality.AUDIO), None)
    if existing is None:
        return results + [
            ModalityResult(
                modality=Modality.AUDIO,
                findings=typed_findings,
                summary=typed_summary,
            )
        ]
    merged = ModalityResult(
        modality=Modality.AUDIO,
        findings=list(existing.findings) + typed_findings,
        summary=f"{existing.summary} | {typed_summary}",
        metadata=existing.metadata,
    )
    return [r for r in results if r.modality != Modality.AUDIO] + [merged]


@app.post("/intake")
async def intake(
    patient_id: str = Form(..., description="Identificador do paciente"),
    data: str | None = Form(
        None,
        description=(
            "JSON com {symptoms, readings, prescriptions} — todos opcionais. "
            "Ex.: {\"symptoms\":\"dor no peito\",\"readings\":[...],"
            "\"prescriptions\":[...]}"
        ),
    ),
    audio: UploadFile | None = File(
        None, description="Áudio da consulta (.wav 16kHz mono ou .mp3)"
    ),
    video: UploadFile | None = File(
        None, description="Vídeo do paciente (.mp4) — análise postural via MediaPipe"
    ),
) -> dict:
    """Captura clínica ponta-a-ponta ("admissão" multimodal).

    Aceita qualquer combinação de:
    - sintomas em texto livre (analisados via Text Analytics)
    - leituras de sinais vitais (RF03)
    - eventos de prescrição (RF03)
    - áudio de consulta (RF02, Azure Speech to Text ou fallback offline)
    - vídeo do paciente (RF01, MediaPipe Pose)

    Executa o pipeline multimodal completo e devolve o relatório com score
    de risco e alerta (se ultrapassar o limiar). Arquivos são gravados em
    diretório temporário e removidos ao final.
    """
    try:
        return await _intake_impl(patient_id, data, audio, video)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - devolvemos detalhe rico ao cliente
        logging.getLogger(__name__).exception("Falha no /intake")
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc().splitlines()[-8:],
            },
        ) from exc


async def _intake_impl(
    patient_id: str,
    data: str | None,
    audio: UploadFile | None,
    video: UploadFile | None,
) -> dict:
    """Corpo real do handler de intake (isolado para tratamento uniforme de erros)."""
    payload: dict = {}
    if data:
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            raise HTTPException(400, f"campo 'data' não é JSON válido: {exc}") from exc
        if not isinstance(payload, dict):
            raise HTTPException(400, "campo 'data' deve ser um objeto JSON")

    try:
        readings = [VitalSignReading(**r) for r in payload.get("readings", [])]
        prescriptions = [
            PrescriptionEvent(**p) for p in payload.get("prescriptions", [])
        ]
    except Exception as exc:
        raise HTTPException(400, f"payload inválido: {exc}") from exc

    symptoms: str | None = payload.get("symptoms")

    with tempfile.TemporaryDirectory(prefix="intake_") as tmp:
        tmp_dir = Path(tmp)
        audio_path: Path | None = None
        video_path: Path | None = None

        if audio is not None and audio.filename:
            audio_path = await _save_upload(audio, tmp_dir)
        if video is not None and video.filename:
            video_path = await _save_upload(video, tmp_dir)

        monitor = _get_monitor()
        report = monitor.run(
            MonitoringInput(
                patient_id=patient_id,
                vitals=_readings_to_df(readings),
                prescriptions=prescriptions,
                audio_path=audio_path,
                video_path=video_path,
            )
        )
        all_results = list(report.modality_results)

        # sintomas digitados vão para a modalidade AUDIO (fala do paciente
        # transcrita antecipadamente pela enfermagem/médico), mesclando com
        # o resultado do áudio se este também foi enviado.
        if symptoms:
            insights = monitor.audio_pipeline.text_analytics.analyze(symptoms)
            typed_findings = [
                Finding(
                    modality=Modality.AUDIO,
                    severity=sev,
                    description=f"Sintoma relatado (digitado): '{term}'",
                    score=0.6,
                    metadata={"term": term, "source": "typed"},
                )
                for term, sev in insights.critical_terms
            ]
            typed_summary = (
                f"Sintomas digitados: sentimento {insights.sentiment} "
                f"({insights.sentiment_score:+.2f}); "
                f"{len(insights.critical_terms)} termo(s) crítico(s)"
            )
            all_results = _merge_or_append_audio(
                all_results, typed_findings, typed_summary
            )

        risk_score, alert = monitor.fusion.fuse(patient_id, all_results)

    return {
        "patient_id": patient_id,
        "generated_at": datetime.now().isoformat(),
        "inputs_received": {
            "symptoms_text": bool(symptoms),
            "vitals_readings": len(readings),
            "prescriptions": len(prescriptions),
            "audio_file": (audio.filename if audio and audio.filename else None),
            "video_file": (video.filename if video and video.filename else None),
        },
        "risk_score": risk_score,
        "alert": alert.model_dump(mode="json") if alert else None,
        "modalities": {
            r.modality.value: {
                "risk_score": r.risk_score,
                "summary": r.summary,
                "findings": [f.model_dump(mode="json") for f in r.findings],
            }
            for r in all_results
        },
    }


# --------------------------------------------------------------------------- #
# Coorte de pacientes — seleção e análise com "amarração" ao ID do paciente
# --------------------------------------------------------------------------- #
def _get_cohort() -> PatientCohort:
    return PatientCohort()


@app.get("/api/patients")
def list_patients() -> dict:
    """Lista os pacientes da coorte persistida (metadados, sem dados brutos)."""
    cohort = _get_cohort()
    if not cohort.available:
        return {"available": False, "patients": []}
    return {
        "available": True,
        "patients": [p.to_dict() for p in cohort.list_patients()],
    }


class PatientAnalyzeRequest(BaseModel):
    """Parâmetros da análise de um paciente da coorte."""

    use_real_video: bool = Field(
        default=False,
        description=(
            "Se True e o paciente tiver vídeo vinculado, processa o arquivo de "
            "vídeo real (MediaPipe + YOLOv8); caso contrário usa a pose "
            "pré-computada (mais rápido)."
        ),
    )


@app.post("/api/patients/{patient_id}/analyze")
def analyze_patient(patient_id: str, req: PatientAnalyzeRequest | None = None) -> dict:
    """Carrega automaticamente os dados vinculados ao paciente e os analisa.

    "Amarração": a partir do ``patient_id``, o sistema busca na coorte os
    sinais vitais, prescrições, transcrição de consulta e sinais de vídeo/pose
    já persistidos e executa o pipeline multimodal completo.
    """
    cohort = _get_cohort()
    record = cohort.get(patient_id)
    if record is None:
        raise HTTPException(404, f"Paciente não encontrado na coorte: {patient_id}")

    use_real_video = bool(req and req.use_real_video)
    data = cohort.load_input(patient_id, use_real_video=use_real_video)

    monitor = _get_monitor()
    report = monitor.run(data)
    payload = report.as_dict()
    payload["patient"] = record.to_dict()
    payload["used_real_video"] = use_real_video
    return payload
