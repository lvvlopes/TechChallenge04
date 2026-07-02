"""Análise de vídeo clínico (RF01).

Análise postural (MediaPipe Pose, como alternativa prática ao OpenPose) e
detecção de objetos/áreas críticas (YOLOv8/ultralytics). Ambas as dependências
são opcionais: sem elas (ou sem GPU), o pipeline opera em modo offline sobre
sinais de pose pré-computados, permitindo demonstração reprodutível.
"""

from __future__ import annotations

from .object_detector import ObjectDetector
from .pose_analyzer import PoseAnalyzer, PoseFrame
from .video_pipeline import VideoAnalysisPipeline

__all__ = [
    "PoseAnalyzer",
    "PoseFrame",
    "ObjectDetector",
    "VideoAnalysisPipeline",
]
