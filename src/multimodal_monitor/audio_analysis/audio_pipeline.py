"""Pipeline de análise de áudio (RF02): áudio → transcrição → insights → achados."""

from __future__ import annotations

from pathlib import Path

from ..config import Settings, get_settings
from ..logging_config import get_logger
from ..schemas import Finding, Modality, ModalityResult, Severity
from .speech_to_text import SpeechToText
from .text_analytics import TextAnalytics

logger = get_logger(__name__)


class AudioAnalysisPipeline:
    """Orquestra transcrição + análise de texto de um áudio de consulta."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.stt = SpeechToText(self.settings)
        self.text_analytics = TextAnalytics(self.settings)

    def process(self, audio_path: str | Path) -> ModalityResult:
        audio_path = Path(audio_path)
        transcription = self.stt.transcribe(audio_path)
        insights = self.text_analytics.analyze(transcription.text)

        findings: list[Finding] = []

        # 1. Termos críticos mencionados na fala do paciente
        for term, severity in insights.critical_terms:
            findings.append(
                Finding(
                    modality=Modality.AUDIO,
                    severity=severity,
                    description=f"Termo crítico mencionado: '{term}'",
                    score=min(0.4 + severity.weight * 0.6, 1.0),
                    metadata={"term": term, "source": transcription.source},
                )
            )

        # 2. Sentimento fortemente negativo pode indicar sofrimento/piora
        if insights.sentiment == "negative" and insights.sentiment_score < -0.4:
            findings.append(
                Finding(
                    modality=Modality.AUDIO,
                    severity=Severity.MEDIUM,
                    description=(
                        "Sentimento negativo acentuado na fala do paciente "
                        f"(score={insights.sentiment_score:.2f})"
                    ),
                    score=min(abs(insights.sentiment_score), 1.0),
                    metadata={"sentiment": insights.sentiment},
                )
            )

        text = transcription.text
        preview = (text[:120] + "…") if len(text) > 120 else text
        summary = (
            f"Transcrição ({transcription.source}): {preview} | "
            f"Sentimento: {insights.sentiment} ({insights.sentiment_score:+.2f}) | "
            f"Termos críticos: {len(insights.critical_terms)}"
        )

        return ModalityResult(
            modality=Modality.AUDIO,
            findings=findings,
            summary=summary,
            metadata={
                "transcript": transcription.text,
                "transcript_source": transcription.source,
                "sentiment": insights.sentiment,
                "sentiment_score": insights.sentiment_score,
            },
        )
