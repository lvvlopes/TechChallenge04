"""Transcrição de áudio via Azure Speech to Text (RF02).

Se as credenciais Azure estiverem ausentes (ou o SDK não estiver instalado),
opera em **modo mock**: procura por um arquivo `.txt` irmão do áudio (mesmo
nome, extensão `.txt`) contendo a transcrição de referência. Isso permite
demonstrar todo o pipeline offline, sem custo de nuvem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..config import Settings, get_settings
from ..logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TranscriptionResult:
    """Resultado da transcrição de um áudio."""

    text: str
    locale: str
    source: str  # "azure" | "mock"
    duration_seconds: float | None = None
    segments: list[dict] = field(default_factory=list)


class SpeechToText:
    """Wrapper de transcrição com degradação graciosa."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @property
    def enabled(self) -> bool:
        return self.settings.azure_speech_enabled

    def transcribe(self, audio_path: str | Path, locale: str | None = None) -> TranscriptionResult:
        """Transcreve um arquivo de áudio para texto."""
        audio_path = Path(audio_path)
        locale = locale or self.settings.default_locale

        if self.enabled:
            try:
                return self._transcribe_azure(audio_path, locale)
            except Exception as exc:  # pragma: no cover - depende de rede/SDK
                logger.warning("Falha no Azure Speech (%s); usando modo mock.", exc)

        return self._transcribe_mock(audio_path, locale)

    # ------------------------------------------------------------------ #
    def _ensure_wav(self, audio_path: Path) -> Path:
        """Garante um arquivo WAV PCM para o Azure Speech SDK.

        O SDK só decodifica WAV nativamente (formatos comprimidos como
        WebM/Opus/MP3 exigem GStreamer). Se o arquivo não for `.wav` e o
        ffmpeg estiver disponível no PATH, converte para WAV 16 kHz mono;
        caso contrário, devolve o caminho original (a transcrição
        provavelmente falhará e o chamador cairá no modo mock).
        """
        if audio_path.suffix.lower() == ".wav":
            return audio_path

        import shutil
        import subprocess
        import tempfile

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            logger.warning(
                "Arquivo %s não é WAV e o ffmpeg não está no PATH — o Azure "
                "Speech pode não conseguir decodificá-lo. Instale o ffmpeg "
                "ou envie WAV 16 kHz mono.",
                audio_path.name,
            )
            return audio_path

        dst = Path(tempfile.gettempdir()) / f"{audio_path.stem}_16k.wav"
        subprocess.run(
            [
                ffmpeg, "-y", "-i", str(audio_path),
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                str(dst),
            ],
            check=True,
            capture_output=True,
        )
        logger.info("Áudio convertido para WAV 16 kHz mono: %s", dst.name)
        return dst

    def _transcribe_azure(self, audio_path: Path, locale: str) -> TranscriptionResult:
        """Transcrição real via Azure Speech SDK (contínua)."""
        import azure.cognitiveservices.speech as speechsdk  # type: ignore

        audio_path = self._ensure_wav(audio_path)

        speech_config = speechsdk.SpeechConfig(
            subscription=self.settings.azure_speech_key,
            region=self.settings.azure_speech_region,
        )
        speech_config.speech_recognition_language = locale
        audio_config = speechsdk.audio.AudioConfig(filename=str(audio_path))
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_config
        )

        segments: list[dict] = []
        done = False

        def _on_recognized(evt: speechsdk.SpeechRecognitionEventArgs) -> None:
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                segments.append({"text": evt.result.text, "offset": evt.result.offset})

        def _on_stopped(_evt: object) -> None:
            nonlocal done
            done = True

        recognizer.recognized.connect(_on_recognized)
        recognizer.session_stopped.connect(_on_stopped)
        recognizer.canceled.connect(_on_stopped)

        recognizer.start_continuous_recognition()
        import time

        while not done:
            time.sleep(0.2)
        recognizer.stop_continuous_recognition()

        text = " ".join(s["text"] for s in segments).strip()
        logger.info("Transcrição Azure concluída (%d segmentos).", len(segments))
        return TranscriptionResult(
            text=text, locale=locale, source="azure", segments=segments
        )

    # ------------------------------------------------------------------ #
    def _transcribe_mock(self, audio_path: Path, locale: str) -> TranscriptionResult:
        """Modo offline: lê transcrição de referência de um `.txt` irmão."""
        transcript_file = audio_path.with_suffix(".txt")
        if transcript_file.exists():
            text = transcript_file.read_text(encoding="utf-8").strip()
            logger.info("Transcrição mock carregada de %s", transcript_file.name)
        else:
            text = ""
            logger.warning(
                "Modo mock: nenhum arquivo de transcrição encontrado para %s. "
                "Crie %s com o texto de referência.",
                audio_path.name,
                transcript_file.name,
            )
        return TranscriptionResult(text=text, locale=locale, source="mock")
