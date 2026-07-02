"""Análise de texto clínico via Azure Language / Text Analytics (RF02).

Extrai:
- **Sentimento** (positivo/neutro/negativo) da fala do paciente.
- **Termos críticos** — sintomas/expressões de risco clínico.

Degradação graciosa: sem credenciais Azure, usa um analisador léxico offline
baseado em dicionários de termos críticos e palavras de polaridade em pt-BR.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from ..config import Settings, get_settings
from ..logging_config import get_logger
from ..schemas import Severity

logger = get_logger(__name__)


# Termos críticos clínicos em pt-BR com severidade associada.
CRITICAL_TERMS: dict[str, Severity] = {
    "falta de ar": Severity.HIGH,
    "dificuldade para respirar": Severity.HIGH,
    "dor no peito": Severity.CRITICAL,
    "aperto no peito": Severity.CRITICAL,
    "desmaio": Severity.HIGH,
    "desmaiei": Severity.HIGH,
    "tontura": Severity.MEDIUM,
    "cansaco": Severity.MEDIUM,
    "cansada": Severity.MEDIUM,
    "cansado": Severity.MEDIUM,
    "fadiga": Severity.MEDIUM,
    "fraqueza": Severity.MEDIUM,
    "palpitacao": Severity.HIGH,
    "sangramento": Severity.HIGH,
    "febre": Severity.MEDIUM,
    "confusao": Severity.HIGH,
    "nao consigo respirar": Severity.CRITICAL,
    "dormencia": Severity.HIGH,
}

_NEG_WORDS = {
    "dor", "ruim", "pior", "piorou", "mal", "difícil", "dificil", "sofrendo",
    "preocupado", "preocupada", "medo", "cansado", "cansada", "fraco", "fraca",
    "triste", "angustia", "angústia",
}
_POS_WORDS = {
    "bem", "melhor", "melhorou", "ótimo", "otimo", "bom", "boa", "tranquilo",
    "tranquila", "aliviado", "aliviada", "estável", "estavel", "recuperando",
}


def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


@dataclass
class TextInsights:
    """Insights extraídos de um texto clínico."""

    sentiment: str  # positive | neutral | negative
    sentiment_score: float  # [-1, 1], negativo = pior
    critical_terms: list[tuple[str, Severity]] = field(default_factory=list)
    source: str = "mock"


class TextAnalytics:
    """Análise de sentimento e termos críticos com degradação graciosa."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @property
    def enabled(self) -> bool:
        return self.settings.azure_language_enabled

    def analyze(self, text: str) -> TextInsights:
        if not text.strip():
            return TextInsights(sentiment="neutral", sentiment_score=0.0, source="empty")

        critical_terms = self._extract_critical_terms(text)

        if self.enabled:
            try:
                sentiment, score = self._sentiment_azure(text)
                return TextInsights(
                    sentiment=sentiment,
                    sentiment_score=score,
                    critical_terms=critical_terms,
                    source="azure",
                )
            except Exception as exc:  # pragma: no cover - depende de rede/SDK
                logger.warning("Falha no Azure Language (%s); usando modo léxico.", exc)

        sentiment, score = self._sentiment_lexical(text)
        return TextInsights(
            sentiment=sentiment,
            sentiment_score=score,
            critical_terms=critical_terms,
            source="mock",
        )

    # ------------------------------------------------------------------ #
    def _extract_critical_terms(self, text: str) -> list[tuple[str, Severity]]:
        norm = _strip_accents(text.lower())
        found: list[tuple[str, Severity]] = []
        for term, severity in CRITICAL_TERMS.items():
            if _strip_accents(term) in norm:
                found.append((term, severity))
        return found

    def _sentiment_lexical(self, text: str) -> tuple[str, float]:
        tokens = re.findall(r"\w+", text.lower())
        neg = sum(1 for t in tokens if t in _NEG_WORDS)
        pos = sum(1 for t in tokens if t in _POS_WORDS)
        total = neg + pos
        if total == 0:
            return "neutral", 0.0
        score = (pos - neg) / total
        if score > 0.2:
            return "positive", score
        if score < -0.2:
            return "negative", score
        return "neutral", score

    def _sentiment_azure(self, text: str) -> tuple[str, float]:
        """Análise de sentimento real via Azure Text Analytics."""
        from azure.ai.textanalytics import TextAnalyticsClient  # type: ignore
        from azure.core.credentials import AzureKeyCredential  # type: ignore

        client = TextAnalyticsClient(
            endpoint=self.settings.azure_language_endpoint,
            credential=AzureKeyCredential(self.settings.azure_language_key),
        )
        result = client.analyze_sentiment(documents=[text], language="pt")[0]
        if result.is_error:  # type: ignore[attr-defined]
            raise RuntimeError("Azure Text Analytics retornou erro")
        scores = result.confidence_scores  # type: ignore[attr-defined]
        signed = float(scores.positive - scores.negative)
        return result.sentiment, signed  # type: ignore[attr-defined]
