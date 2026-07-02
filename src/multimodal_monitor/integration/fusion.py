"""Fusão multimodal (late fusion) → score de risco global e alerta.

Estratégia
----------
Cada modalidade produz um :class:`ModalityResult` com um ``risk_score`` em
[0,1]. A fusão combina esses scores com uma média ponderada e aplica um
**bônus de corroboração**: quando múltiplas modalidades concordam num risco
elevado, a confiança clínica aumenta (o todo é maior que a soma das partes).

O score final é mapeado para uma :class:`Severity`, que dispara (ou não) um
:class:`Alert` para a equipe médica.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..logging_config import get_logger
from ..schemas import Alert, Finding, Modality, ModalityResult, Severity

logger = get_logger(__name__)

# Peso clínico relativo de cada modalidade na fusão.
DEFAULT_WEIGHTS: dict[Modality, float] = {
    Modality.VITALS: 1.0,
    Modality.AUDIO: 0.6,
    Modality.VIDEO: 0.7,
    Modality.PRESCRIPTION: 0.9,
    Modality.MOVEMENT: 0.6,
}

# Faixas de score → severidade do alerta.
SEVERITY_BANDS: list[tuple[float, Severity]] = [
    (0.85, Severity.CRITICAL),
    (0.6, Severity.HIGH),
    (0.35, Severity.MEDIUM),
    (0.15, Severity.LOW),
]


@dataclass
class MultimodalFusion:
    """Combina resultados de múltiplas modalidades em um score de risco único."""

    weights: dict[Modality, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    # score mínimo para emitir alerta
    alert_threshold: float = 0.35
    # bônus por modalidade adicional que concorda em risco relevante
    corroboration_bonus: float = 0.1
    # risco mínimo de uma modalidade para contar como corroboração
    corroboration_min: float = 0.25

    def fuse(self, patient_id: str, results: list[ModalityResult]) -> tuple[float, Alert | None]:
        """Funde os resultados e retorna ``(risk_score, alerta_ou_None)``."""
        active = [r for r in results if r.risk_score > 0]
        if not active:
            return 0.0, None

        weighted_sum = 0.0
        weight_total = 0.0
        for r in active:
            w = self.weights.get(r.modality, 0.5)
            weighted_sum += r.risk_score * w
            weight_total += w
        base_score = weighted_sum / weight_total if weight_total else 0.0

        # corroboração: nº de modalidades com risco relevante
        corroborating = sum(1 for r in active if r.risk_score >= self.corroboration_min)
        bonus = self.corroboration_bonus * max(0, corroborating - 1)
        risk_score = float(min(base_score + bonus, 1.0))

        severity = self._score_to_severity(risk_score)
        if risk_score < self.alert_threshold:
            logger.debug("Score %.2f abaixo do limiar; sem alerta.", risk_score)
            return risk_score, None

        alert = self._build_alert(patient_id, risk_score, severity, active)
        return risk_score, alert

    # ------------------------------------------------------------------ #
    @staticmethod
    def _score_to_severity(score: float) -> Severity:
        for threshold, severity in SEVERITY_BANDS:
            if score >= threshold:
                return severity
        return Severity.INFO

    def _build_alert(
        self,
        patient_id: str,
        risk_score: float,
        severity: Severity,
        results: list[ModalityResult],
    ) -> Alert:
        # coleta os achados mais graves de cada modalidade
        top_findings: list[Finding] = []
        for r in results:
            top = sorted(r.findings, key=lambda f: f.severity.weight * f.score, reverse=True)
            top_findings.extend(top[:2])
        top_findings.sort(key=lambda f: f.severity.weight * f.score, reverse=True)

        modalities = [r.modality for r in results]
        reasons = "; ".join(f.description for f in top_findings[:4])
        title = self._title_for(severity, modalities)
        message = (
            f"Sinais de risco corroborados por {len(modalities)} modalidade(s) "
            f"({', '.join(m.value for m in modalities)}). Principais achados: {reasons}."
        )

        return Alert(
            patient_id=patient_id,
            severity=severity,
            risk_score=risk_score,
            title=title,
            message=message,
            contributing_findings=top_findings[:6],
            modalities=modalities,
        )

    @staticmethod
    def _title_for(severity: Severity, modalities: list[Modality]) -> str:
        if severity is Severity.CRITICAL:
            return "Risco clínico CRÍTICO — intervenção imediata"
        if severity is Severity.HIGH:
            return "Risco clínico ELEVADO — avaliação prioritária"
        if severity is Severity.MEDIUM:
            return "Atenção — sinais de alerta detectados"
        return "Observação — variação leve detectada"
