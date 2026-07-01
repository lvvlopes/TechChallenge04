"""Modelos de domínio compartilhados entre os módulos multimodais.

Estes schemas (Pydantic) formam o *contrato* entre as três modalidades
(vídeo, áudio, texto/sinais vitais) e a camada de fusão + alertas.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Modality(str, Enum):
    """Modalidade de origem de um sinal/observação."""

    VIDEO = "video"
    AUDIO = "audio"
    VITALS = "vitals"
    PRESCRIPTION = "prescription"
    MOVEMENT = "movement"


class Severity(str, Enum):
    """Severidade de um achado clínico ou alerta."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def weight(self) -> float:
        """Peso numérico usado na fusão multimodal."""
        return {
            Severity.INFO: 0.0,
            Severity.LOW: 0.25,
            Severity.MEDIUM: 0.5,
            Severity.HIGH: 0.75,
            Severity.CRITICAL: 1.0,
        }[self]


class Finding(BaseModel):
    """Um achado individual detectado por uma modalidade.

    Ex.: "postura fora do padrão no quadro 512", "queda de SpO2 detectada",
    "termo crítico 'falta de ar' mencionado".
    """

    modality: Modality
    severity: Severity
    description: str
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Confiança/intensidade [0,1]")
    timestamp: datetime = Field(default_factory=_utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModalityResult(BaseModel):
    """Resultado agregado do processamento de uma modalidade."""

    modality: Modality
    findings: list[Finding] = Field(default_factory=list)
    summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def max_severity(self) -> Severity:
        """Maior severidade entre os achados (INFO se não houver achados)."""
        if not self.findings:
            return Severity.INFO
        return max(self.findings, key=lambda f: f.severity.weight).severity

    @property
    def risk_score(self) -> float:
        """Score de risco da modalidade em [0,1].

        Combina a severidade e a confiança dos achados, dando mais peso ao
        achado mais grave e amortecendo os demais (evita saturação por volume).
        """
        if not self.findings:
            return 0.0
        contributions = sorted(
            (f.severity.weight * max(f.score, 0.1) for f in self.findings),
            reverse=True,
        )
        # peso decrescente: o achado mais grave domina; os demais somam pouco
        total = 0.0
        for i, c in enumerate(contributions):
            total += c * (0.5**i)
        return float(min(total, 1.0))


class Alert(BaseModel):
    """Alerta clínico gerado a partir da fusão multimodal."""

    patient_id: str
    severity: Severity
    risk_score: float = Field(ge=0.0, le=1.0)
    title: str
    message: str
    contributing_findings: list[Finding] = Field(default_factory=list)
    modalities: list[Modality] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=_utcnow)

    def to_human(self) -> str:
        """Renderiza o alerta em texto legível para a equipe médica."""
        head = f"[{self.severity.value.upper()}] Paciente {self.patient_id} — {self.title}"
        body = self.message
        score = f"Score de risco: {self.risk_score:.2f}"
        mods = "Modalidades: " + ", ".join(m.value for m in self.modalities)
        return "\n".join([head, body, score, mods])


class VitalSignReading(BaseModel):
    """Uma leitura instantânea de sinais vitais."""

    timestamp: datetime
    heart_rate: float | None = None  # bpm
    systolic_bp: float | None = None  # mmHg
    diastolic_bp: float | None = None  # mmHg
    spo2: float | None = None  # %
    respiratory_rate: float | None = None  # rpm
    temperature: float | None = None  # °C
