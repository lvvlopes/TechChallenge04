"""Análise postural quadro a quadro (RF01).

Extrai landmarks corporais com MediaPipe Pose e deriva, por quadro:

- ``movement_index`` — magnitude do movimento entre quadros consecutivos [0,1];
- ``trunk_angle`` — ângulo do tronco em relação à vertical (graus).

Sem MediaPipe/OpenCV instalados, :func:`PoseAnalyzer.analyze_video` levanta
:class:`RuntimeError`; use então o modo offline do :class:`VideoAnalysisPipeline`
com sinais de pose pré-computados.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from ..logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PoseFrame:
    """Métricas de pose de um único quadro de vídeo."""

    frame_index: int
    timestamp_s: float
    movement_index: float
    trunk_angle: float | None = None


def _vision_available() -> bool:
    try:
        import cv2  # noqa: F401
        import mediapipe  # noqa: F401

        return True
    except ImportError:
        return False


class PoseAnalyzer:
    """Extrator de métricas posturais baseado em MediaPipe Pose."""

    def __init__(self, sample_rate: int = 5) -> None:
        """``sample_rate``: processa 1 a cada N quadros (desempenho)."""
        self.sample_rate = max(1, sample_rate)

    @property
    def available(self) -> bool:
        return _vision_available()

    def analyze_video(self, video_path: str | Path) -> list[PoseFrame]:
        """Processa um vídeo e retorna métricas posturais por quadro amostrado."""
        if not self.available:
            raise RuntimeError(
                "MediaPipe/OpenCV não instalados. Instale com "
                "`pip install opencv-python mediapipe` ou use o modo offline."
            )

        import cv2  # type: ignore
        import mediapipe as mp  # type: ignore

        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        pose = mp.solutions.pose.Pose(model_complexity=1, min_detection_confidence=0.5)
        frames: list[PoseFrame] = []
        prev_landmarks = None
        idx = 0

        try:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % self.sample_rate != 0:
                    idx += 1
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)
                if result.pose_landmarks:
                    lm = [(p.x, p.y) for p in result.pose_landmarks.landmark]
                    movement = self._movement(prev_landmarks, lm)
                    trunk = self._trunk_angle(result.pose_landmarks.landmark)
                    frames.append(
                        PoseFrame(
                            frame_index=idx,
                            timestamp_s=idx / fps,
                            movement_index=movement,
                            trunk_angle=trunk,
                        )
                    )
                    prev_landmarks = lm
                idx += 1
        finally:
            cap.release()
            pose.close()

        logger.info("Pose extraída de %d quadros amostrados.", len(frames))
        return frames

    # ------------------------------------------------------------------ #
    @staticmethod
    def _movement(prev: list[tuple[float, float]] | None, cur: list[tuple[float, float]]) -> float:
        if prev is None or len(prev) != len(cur):
            return 0.0
        total = sum(math.dist(p, c) for p, c in zip(prev, cur, strict=True))
        # normaliza pela quantidade de landmarks e satura em [0,1]
        return float(min(total / len(cur) * 10.0, 1.0))

    @staticmethod
    def _trunk_angle(landmarks: object) -> float | None:
        """Ângulo do tronco (ombros→quadris) em relação à vertical, em graus."""
        try:
            import mediapipe as mp  # type: ignore

            pl = mp.solutions.pose.PoseLandmark
            ls, rs = landmarks[pl.LEFT_SHOULDER], landmarks[pl.RIGHT_SHOULDER]
            lh, rh = landmarks[pl.LEFT_HIP], landmarks[pl.RIGHT_HIP]
            shoulder = ((ls.x + rs.x) / 2, (ls.y + rs.y) / 2)
            hip = ((lh.x + rh.x) / 2, (lh.y + rh.y) / 2)
            dx, dy = shoulder[0] - hip[0], shoulder[1] - hip[1]
            angle = abs(math.degrees(math.atan2(dx, -dy)))
            return float(angle)
        except Exception:  # pragma: no cover
            return None
