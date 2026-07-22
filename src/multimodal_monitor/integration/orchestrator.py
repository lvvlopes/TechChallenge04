"""Orquestração multimodal ponta-a-ponta (RF01+RF02+RF03).

O :class:`PatientMonitor` recebe as entradas disponíveis de um paciente
(sinais vitais, áudio de consulta, sinais de pose de vídeo, evolução de
prescrições), executa os pipelines de cada modalidade, funde os resultados e
despacha um alerta quando o risco ultrapassa o limiar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..anomaly_detection.prescriptions import PrescriptionAnomalyDetector, PrescriptionEvent
from ..anomaly_detection.vitals import VitalsAnomalyDetector
from ..audio_analysis.audio_pipeline import AudioAnalysisPipeline
from ..config import Settings, get_settings
from ..logging_config import get_logger
from ..schemas import Alert, Modality, ModalityResult
from ..video_analysis.pose_analyzer import PoseFrame
from ..video_analysis.video_pipeline import VideoAnalysisPipeline
from .alerts import AlertManager
from .fusion import MultimodalFusion

logger = get_logger(__name__)


@dataclass
class MonitoringInput:
    """Conjunto de entradas multimodais de um paciente para um ciclo de análise."""

    patient_id: str
    vitals: pd.DataFrame | None = None
    audio_path: str | Path | None = None
    pose_frames: list[PoseFrame] | None = None
    video_path: str | Path | None = None
    prescriptions: list[PrescriptionEvent] = field(default_factory=list)


@dataclass
class MonitoringReport:
    """Resultado consolidado de um ciclo de monitoramento."""

    patient_id: str
    risk_score: float
    alert: Alert | None
    modality_results: list[ModalityResult]
    generated_at: datetime = field(default_factory=lambda: datetime.now())

    def as_dict(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "risk_score": self.risk_score,
            "alert": self.alert.model_dump(mode="json") if self.alert else None,
            "modalities": {
                r.modality.value: {
                    "risk_score": r.risk_score,
                    "summary": r.summary,
                    "findings": [f.model_dump(mode="json") for f in r.findings],
                    "metadata": r.metadata,
                }
                for r in self.modality_results
            },
            "generated_at": self.generated_at.isoformat(),
        }


class PatientMonitor:
    """Fachada de alto nível que executa o pipeline multimodal completo."""

    def __init__(self, settings: Settings | None = None, dispatch_alerts: bool = True) -> None:
        self.settings = settings or get_settings()
        self.vitals_detector = VitalsAnomalyDetector(
            contamination=self.settings.anomaly_contamination,
            zscore_threshold=self.settings.vitals_zscore_threshold,
        )
        self.prescription_detector = PrescriptionAnomalyDetector()
        self.audio_pipeline = AudioAnalysisPipeline(self.settings)
        self.video_pipeline = VideoAnalysisPipeline()
        self.fusion = MultimodalFusion()
        self.alert_manager = AlertManager(self.settings)
        self.dispatch_alerts = dispatch_alerts

    def run(self, data: MonitoringInput) -> MonitoringReport:
        """Executa um ciclo de monitoramento para um paciente."""
        results: list[ModalityResult] = []

        # RF03 — sinais vitais
        if data.vitals is not None and not data.vitals.empty:
            findings = self.vitals_detector.detect(data.vitals)
            results.append(
                ModalityResult(
                    modality=Modality.VITALS,
                    findings=findings,
                    summary=f"{len(findings)} anomalia(s) em sinais vitais.",
                )
            )

        # RF03 — prescrições
        if data.prescriptions:
            findings = self.prescription_detector.detect(data.prescriptions)
            results.append(
                ModalityResult(
                    modality=Modality.PRESCRIPTION,
                    findings=findings,
                    summary=f"{len(findings)} anomalia(s) em prescrições.",
                )
            )

        # RF02 — áudio (protegido contra qualquer falha de decodificação/SDK)
        if data.audio_path:
            try:
                results.append(self.audio_pipeline.process(data.audio_path))
            except Exception as exc:
                logger.warning(
                    "Análise de áudio falhou (%s: %s); prosseguindo sem essa modalidade.",
                    type(exc).__name__,
                    exc,
                )

        # RF01 — vídeo (online se houver arquivo; offline se houver pose pré-computada)
        # Blindado contra qualquer exceção (OpenCV pode falhar em WebM/codecs
        # desconhecidos, MediaPipe pode explodir em quadros corrompidos, etc.)
        if data.video_path:
            try:
                # análise pesada isolada em subprocesso (evita colisões de
                # protobuf/torch/mediapipe dentro do servidor web)
                results.append(self.video_pipeline.process_isolated(data.video_path))
            except Exception as exc:
                logger.warning(
                    "Análise de vídeo falhou (%s: %s); prosseguindo sem essa modalidade.",
                    type(exc).__name__,
                    exc,
                )
        elif data.pose_frames:
            results.append(self.video_pipeline.process_pose_frames(data.pose_frames))

        risk_score, alert = self.fusion.fuse(data.patient_id, results)

        if alert and self.dispatch_alerts:
            self.alert_manager.dispatch(alert)

        return MonitoringReport(
            patient_id=data.patient_id,
            risk_score=risk_score,
            alert=alert,
            modality_results=results,
        )
