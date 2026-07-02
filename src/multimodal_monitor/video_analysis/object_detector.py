"""Detecção de objetos / áreas críticas em vídeo com YOLOv8 (RF01).

Usa `ultralytics` quando disponível. Objetos relevantes num contexto clínico
podem indicar risco (ex.: paciente próximo à borda do leito, ausência de
equipamento esperado). Sem a dependência, os métodos degradam para no-op.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class Detection:
    """Uma detecção de objeto em um quadro."""

    frame_index: int
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 (normalizado)


class ObjectDetector:
    """Wrapper de detecção de objetos baseado em YOLOv8."""

    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.35) -> None:
        self.model_name = model_name
        self.conf = conf
        self._model = None

    @property
    def available(self) -> bool:
        try:
            import ultralytics  # noqa: F401

            return True
        except ImportError:
            return False

    def _load(self) -> object | None:
        if self._model is not None:
            return self._model
        if not self.available:
            return None
        from ultralytics import YOLO  # type: ignore

        self._model = YOLO(self.model_name)
        return self._model

    def detect_video(self, video_path: str | Path, sample_rate: int = 15) -> list[Detection]:
        """Detecta objetos em quadros amostrados do vídeo."""
        model = self._load()
        if model is None:
            logger.warning("ultralytics ausente; detecção de objetos desabilitada.")
            return []

        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(video_path))
        detections: list[Detection] = []
        idx = 0
        try:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % max(1, sample_rate) == 0:
                    results = model.predict(frame, conf=self.conf, verbose=False)
                    h, w = frame.shape[:2]
                    for r in results:
                        for box in r.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            detections.append(
                                Detection(
                                    frame_index=idx,
                                    label=model.names[int(box.cls[0])],
                                    confidence=float(box.conf[0]),
                                    bbox=(x1 / w, y1 / h, x2 / w, y2 / h),
                                )
                            )
                idx += 1
        finally:
            cap.release()
        logger.info("Detecções de objetos: %d", len(detections))
        return detections
