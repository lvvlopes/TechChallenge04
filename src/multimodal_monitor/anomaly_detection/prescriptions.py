"""Detecção de anomalias na evolução de prescrições (RF03).

Analisa uma sequência temporal de itens prescritos e sinaliza alterações
inesperadas no tratamento, tais como:

- Saltos de dose acima de um fator seguro entre prescrições consecutivas.
- Doses acima da dose máxima diária de referência do medicamento.
- Interações medicamentosas conhecidas (lista curada, ilustrativa).
- Reintrodução de medicamento previamente suspenso.

As bases (doses máximas / interações) são ILUSTRATIVAS e não substituem uma
base farmacológica oficial; servem para demonstrar o mecanismo de detecção.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel

from ..schemas import Finding, Modality, Severity

# Dose máxima diária de referência (mg) — ILUSTRATIVO.
MAX_DAILY_DOSE_MG: dict[str, float] = {
    "paracetamol": 4000,
    "dipirona": 4000,
    "ibuprofeno": 2400,
    "morfina": 200,
    "warfarina": 10,
    "insulina": 100,  # unidades, tratado como mg para simplificar o exemplo
    "amoxicilina": 3000,
    "furosemida": 600,
}

# Pares de interação medicamentosa conhecidos — ILUSTRATIVO.
KNOWN_INTERACTIONS: set[frozenset[str]] = {
    frozenset({"warfarina", "ibuprofeno"}),  # risco de sangramento
    frozenset({"warfarina", "dipirona"}),
    frozenset({"morfina", "insulina"}),  # ilustrativo
}

# Fator máximo de aumento de dose considerado "esperado" entre prescrições.
SAFE_DOSE_JUMP_FACTOR = 2.0


class PrescriptionEvent(BaseModel):
    """Um item de prescrição em um instante do tempo."""

    timestamp: datetime
    drug: str
    dose_mg: float
    action: str = "prescribe"  # prescribe | suspend | resume


@dataclass
class PrescriptionAnomalyDetector:
    """Detector baseado em regras clínicas sobre a evolução de prescrições."""

    safe_jump_factor: float = SAFE_DOSE_JUMP_FACTOR

    def detect(self, events: list[PrescriptionEvent]) -> list[Finding]:
        events = sorted(events, key=lambda e: e.timestamp)
        findings: list[Finding] = []

        last_dose: dict[str, float] = {}
        suspended: set[str] = set()
        active: set[str] = set()

        for ev in events:
            drug = ev.drug.lower().strip()

            findings.extend(self._check_max_dose(ev, drug))
            findings.extend(self._check_dose_jump(ev, drug, last_dose))
            findings.extend(self._check_resume(ev, drug, suspended))

            # atualizar estado
            if ev.action == "suspend":
                suspended.add(drug)
                active.discard(drug)
            else:
                suspended.discard(drug)
                active.add(drug)
                last_dose[drug] = ev.dose_mg

            findings.extend(self._check_interactions(ev, drug, active))

        return findings

    # ------------------------------------------------------------------ #
    def _check_max_dose(self, ev: PrescriptionEvent, drug: str) -> list[Finding]:
        max_dose = MAX_DAILY_DOSE_MG.get(drug)
        if max_dose is None or ev.dose_mg <= max_dose:
            return []
        ratio = ev.dose_mg / max_dose
        return [
            Finding(
                modality=Modality.PRESCRIPTION,
                severity=Severity.CRITICAL if ratio > 1.5 else Severity.HIGH,
                description=(
                    f"Dose de {drug} ({ev.dose_mg:.0f} mg) acima da máxima "
                    f"diária de referência ({max_dose:.0f} mg)"
                ),
                score=min(ratio / 2.0, 1.0),
                timestamp=ev.timestamp,
                metadata={"drug": drug, "dose_mg": ev.dose_mg, "rule": "max_dose"},
            )
        ]

    def _check_dose_jump(
        self, ev: PrescriptionEvent, drug: str, last_dose: dict[str, float]
    ) -> list[Finding]:
        prev = last_dose.get(drug)
        if prev is None or prev <= 0 or ev.action == "suspend":
            return []
        factor = ev.dose_mg / prev
        if factor <= self.safe_jump_factor:
            return []
        return [
            Finding(
                modality=Modality.PRESCRIPTION,
                severity=Severity.HIGH if factor > 3 else Severity.MEDIUM,
                description=(
                    f"Aumento abrupto de dose de {drug}: {prev:.0f} → "
                    f"{ev.dose_mg:.0f} mg ({factor:.1f}x)"
                ),
                score=min((factor - 1) / 4.0, 1.0),
                timestamp=ev.timestamp,
                metadata={"drug": drug, "factor": factor, "rule": "dose_jump"},
            )
        ]

    def _check_resume(
        self, ev: PrescriptionEvent, drug: str, suspended: set[str]
    ) -> list[Finding]:
        if drug not in suspended or ev.action == "suspend":
            return []
        return [
            Finding(
                modality=Modality.PRESCRIPTION,
                severity=Severity.LOW,
                description=f"Reintrodução de {drug}, previamente suspenso",
                score=0.3,
                timestamp=ev.timestamp,
                metadata={"drug": drug, "rule": "resume"},
            )
        ]

    def _check_interactions(
        self, ev: PrescriptionEvent, drug: str, active: set[str]
    ) -> list[Finding]:
        findings: list[Finding] = []
        for other in active:
            if other == drug:
                continue
            if frozenset({drug, other}) in KNOWN_INTERACTIONS:
                findings.append(
                    Finding(
                        modality=Modality.PRESCRIPTION,
                        severity=Severity.HIGH,
                        description=(
                            f"Interação medicamentosa conhecida: {drug} + {other}"
                        ),
                        score=0.8,
                        timestamp=ev.timestamp,
                        metadata={"pair": sorted([drug, other]), "rule": "interaction"},
                    )
                )
        return findings
