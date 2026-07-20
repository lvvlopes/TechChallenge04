"""Testes da análise de áudio/texto (RF02) em modo offline."""

from __future__ import annotations

from multimodal_monitor.audio_analysis.text_analytics import TextAnalytics
from multimodal_monitor.schemas import Severity


def test_extracts_critical_terms_with_accents() -> None:
    ta = TextAnalytics()
    insights = ta.analyze("Estou com muita falta de ar e dor no peito.")
    terms = {t for t, _ in insights.critical_terms}
    assert "falta de ar" in terms
    assert "dor no peito" in terms
    assert any(sev is Severity.CRITICAL for _, sev in insights.critical_terms)


def test_negative_sentiment_detected() -> None:
    ta = TextAnalytics()
    insights = ta.analyze("Estou muito mal, com dor e cansado, me sinto pior hoje.")
    assert insights.sentiment == "negative"
    assert insights.sentiment_score < 0


def test_positive_sentiment_detected() -> None:
    ta = TextAnalytics()
    insights = ta.analyze("Estou bem melhor, tranquilo e me recuperando muito bem.")
    assert insights.sentiment == "positive"


def test_empty_text_is_neutral() -> None:
    insights = TextAnalytics().analyze("   ")
    assert insights.sentiment == "neutral"
    assert insights.critical_terms == []


def test_ensure_wav_passthrough_for_wav(tmp_path) -> None:
    """Arquivos .wav não são convertidos (retorna o próprio caminho)."""
    from multimodal_monitor.audio_analysis.speech_to_text import SpeechToText

    p = tmp_path / "consulta.wav"
    p.write_bytes(b"RIFF....WAVE")
    assert SpeechToText()._ensure_wav(p) == p


def test_ensure_wav_without_ffmpeg_returns_original(tmp_path, monkeypatch) -> None:
    """Sem ffmpeg no PATH, arquivos não-WAV passam adiante inalterados."""
    import shutil

    from multimodal_monitor.audio_analysis.speech_to_text import SpeechToText

    monkeypatch.setattr(shutil, "which", lambda _cmd: None)
    p = tmp_path / "gravacao.webm"
    p.write_bytes(b"\x1aE\xdf\xa3")
    assert SpeechToText()._ensure_wav(p) == p
