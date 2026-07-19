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


def _load_pose_module() -> object | None:
    """Tenta carregar ``mediapipe.solutions.pose`` por várias rotas.

    Diferentes versões / builds do MediaPipe expõem o submódulo por caminhos
    diferentes (lazy loading, import direto, atributo do pacote raiz).
    Tentamos os três em ordem e devolvemos o primeiro que funcionar — ou
    ``None`` se nenhum estiver disponível.
    """
    # 1. caminho canônico moderno
    try:
        from mediapipe.python.solutions import pose as mp_pose

        return mp_pose
    except Exception:  # noqa: BLE001 — qualquer falha de import/init
        pass
    # 2. import direto do submódulo por atributo
    try:
        import mediapipe.solutions.pose as mp_pose  # type: ignore

        return mp_pose
    except Exception:  # noqa: BLE001
        pass
    # 3. lazy: força carga do submódulo via pacote raiz
    try:
        import importlib

        import mediapipe  # noqa: F401

        return importlib.import_module("mediapipe.solutions.pose")
    except Exception:  # noqa: BLE001
        return None


def _vision_available() -> bool:
    """Checa se OpenCV + MediaPipe estão realmente utilizáveis.

    Não basta `import mediapipe` — algumas versões (0.10.9+ em Python 3.12)
    trazem o pacote mas não expõem o submódulo ``solutions.pose`` pelas rotas
    convencionais. Verificamos até o ponto de uso real.
    """
    try:
        import cv2  # noqa: F401
    except ImportError:
        return False
    return _load_pose_module() is not None


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

        mp_pose = _load_pose_module()
        if mp_pose is None:
            raise RuntimeError(
                "Não foi possível carregar mediapipe.solutions.pose. "
                "Sua versão do MediaPipe pode estar incompatível — tente "
                "`pip install 'mediapipe>=0.10.11,<0.11'`."
            )

        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5)
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
            mp_pose = _load_pose_module()
            if mp_pose is None:
                return None
            pl = mp_pose.PoseLandmark
            ls, rs = landmarks[pl.LEFT_SHOULDER], landmarks[pl.RIGHT_SHOULDER]
            lh, rh = landmarks[pl.LEFT_HIP], landmarks[pl.RIGHT_HIP]
            shoulder = ((ls.x + rs.x) / 2, (ls.y + rs.y) / 2)
            hip = ((lh.x + rh.x) / 2, (lh.y + rh.y) / 2)
            dx, dy = shoulder[0] - hip[0], shoulder[1] - hip[1]
            angle = abs(math.degrees(math.atan2(dx, -dy)))
            return float(angle)
        except Exception:  # pragma: no cover
            return None
