"""Análise de áudio de consultas médicas (RF02).

Transcrição via Azure Speech to Text e análise de texto (sentimento + termos
críticos) via Azure Language / Text Analytics, com degradação graciosa para um
modo offline (mock) quando as credenciais Azure não estão configuradas.
"""

from __future__ import annotations

from .audio_pipeline import AudioAnalysisPipeline
from .speech_to_text import SpeechToText, TranscriptionResult
from .text_analytics import TextAnalytics, TextInsights

__all__ = [
    "AudioAnalysisPipeline",
    "SpeechToText",
    "TranscriptionResult",
    "TextAnalytics",
    "TextInsights",
]
