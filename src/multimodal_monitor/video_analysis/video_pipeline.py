"""Pipeline de análise de vídeo (RF01): vídeo → pose/objetos → achados.

Dois modos:

- **online**: processa um arquivo de vídeo real (requer MediaPipe/OpenCV e,
  opcionalmente, ultralytics/YOLOv8);
- **offline**: recebe uma lista de :class:`PoseFrame` pré-computados (ex.:
  carregada de um JSON), permitindo demonstração sem dependências pesadas.

Sobre os sinais posturais aplica-se detecção de anomalias de movimentação e
uma regra de desvio postural (inclinação de tronco acentuada).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..anomaly_detection.movement import MovementAnomalyDetector
from ..logging_config import get_logger
from ..schemas import Finding, Modality, ModalityResult, Severity
from .object_detector import ObjectDetector
from .pose_analyzer import PoseAnalyzer, PoseFrame

logger = get_logger(__name__)

# Inclinação de tronco (graus) a partir da qual se considera desvio postural.
TRUNK_ANGLE_WARN = 35.0
TRUNK_ANGLE_CRITICAL = 55.0


class VideoAnalysisPipeline:
    """Orquestra análise postural e de objetos de um vídeo clínico."""

    def __init__(
        self,
        sample_rate: int = 5,
        enable_object_detection: bool = True,
    ) -> None:
        self.pose_analyzer = PoseAnalyzer(sample_rate=sample_rate)
        self.object_detector = ObjectDetector() if enable_object_detection else None
        self.movement_detector = MovementAnomalyDetector()

    # ------------------------------------------------------------------ #
    def process(self, video_path: str | Path) -> ModalityResult:
        """Modo online: processa um vídeo real."""
        frames = self.pose_analyzer.analyze_video(video_path)
        detections = []
        if self.object_detector and self.object_detector.available:
            detections = self.object_detector.detect_video(video_path)
        return self._build_result(frames, source="video", extra={"detections": len(detections)})

    def process_pose_frames(self, frames: list[PoseFrame]) -> ModalityResult:
        """Modo offline: processa sinais de pose pré-computados."""
        return self._build_result(frames, source="precomputed_pose")

    # ------------------------------------------------------------------ #
    def _build_result(
        self, frames: list[PoseFrame], source: str, extra: dict | None = None
    ) -> ModalityResult:
        findings: list[Finding] = []
        if not frames:
            return ModalityResult(
                modality=Modality.VIDEO,
                summary="Nenhum quadro com pose detectada.",
                metadata={"source": source},
            )

        base_time = datetime.now(timezone.utc)
        timestamps = [base_time + timedelta(seconds=f.timestamp_s) for f in frames]

        # 1. Anomalias de movimentação (imobilidade / picos)
        movement_index = [f.movement_index for f in frames]
        movement_findings = self.movement_detector.detect(timestamps, movement_index)
        # reclassifica a modalidade para VIDEO (origem: análise postural)
        for mf in movement_findings:
            findings.append(mf.model_copy(update={"modality": Modality.VIDEO}))

        # 2. Desvio postural (inclinação de tronco)
        findings.extend(self._posture_findings(frames, timestamps))

        summary = (
            f"Pose analisada em {len(frames)} quadros (fonte: {source}). "
            f"Achados: {len(findings)}."
        )
        meta = {"source": source, "frames": len(frames)}
        if extra:
            meta.update(extra)
        return ModalityResult(
            modality=Modality.VIDEO, findings=findings, summary=summary, metadata=meta
        )

    def _posture_findings(
        self, frames: list[PoseFrame], timestamps: list[datetime]
    ) -> list[Finding]:
        findings: list[Finding] = []
        for frame, ts in zip(frames, timestamps, strict=True):
            angle = frame.trunk_angle
            if angle is None or angle < TRUNK_ANGLE_WARN:
                continue
            severity = Severity.HIGH if angle >= TRUNK_ANGLE_CRITICAL else Severity.MEDIUM
            findings.append(
                Finding(
                    modality=Modality.VIDEO,
                    severity=severity,
                    description=(
                        f"Desvio postural: inclinação de tronco de {angle:.0f}° "
                        "(fora do padrão esperado)"
                    ),
                    score=min(angle / 90.0, 1.0),
                    timestamp=ts,
                    metadata={"trunk_angle": angle, "frame": frame.frame_index},
                )
            )
        return findings
