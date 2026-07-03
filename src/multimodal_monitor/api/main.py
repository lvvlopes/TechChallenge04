"""API REST para monitoramento multimodal near real-time.

Endpoints
---------
- ``GET  /health``            — status e capacidades disponíveis (Azure/visão).
- ``POST /monitor/vitals``    — analisa um lote de leituras de sinais vitais.
- ``POST /monitor/prescriptions`` — analisa uma evolução de prescrições.
- ``POST /monitor``           — ciclo multimodal (vitais + prescrições + texto).

Execução: ``uvicorn multimodal_monitor.api.main:app --reload``
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ..anomaly_detection.prescriptions import PrescriptionEvent
from ..config import get_settings
from ..integration.orchestrator import MonitoringInput, PatientMonitor
from ..schemas import VitalSignReading

app = FastAPI(
    title="Monitoramento Multimodal de Pacientes",
    description=(
        "IA para monitoramento contínuo de pacientes (áudio, vídeo, texto) — FIAP TC Fase 4."
    ),
    version="0.1.0",
)


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
        from ..schemas import Finding, Modality, ModalityResult

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
            summary=f"Sentimento: {insights.sentiment} ({insights.sentiment_score:+.2f})",
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
            }
            for r in results
        },
    }
