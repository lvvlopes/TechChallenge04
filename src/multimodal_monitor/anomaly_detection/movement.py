"""Detecção de anomalias em padrões de movimentação do paciente (RF03).

Recebe uma série temporal de um índice de atividade/movimento (ex.: derivado
da análise de vídeo por pose, ou de um acelerômetro de leito) e sinaliza:

- **Imobilidade prolongada** — risco de úlcera por pressão / TVP.
- **Agitação/queda** — pico abrupto de movimento (possível queda ou crise).

O índice de movimento é um escalar em [0, 1] por instante de tempo.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from ..schemas import Finding, Modality, Severity


@dataclass
class MovementAnomalyDetector:
    """Detector de anomalias sobre um índice de movimento temporal."""

    immobility_threshold: float = 0.05
    immobility_min_steps: int = 12  # nº de amostras consecutivas de baixa atividade
    spike_zscore: float = 3.5

    def detect(
        self, timestamps: list[datetime], movement_index: list[float]
    ) -> list[Finding]:
        if len(movement_index) < 3:
            return []
        arr = np.asarray(movement_index, dtype=float)
        findings: list[Finding] = []
        findings.extend(self._detect_immobility(timestamps, arr))
        findings.extend(self._detect_spikes(timestamps, arr))
        return findings

    def _detect_immobility(
        self, timestamps: list[datetime], arr: np.ndarray
    ) -> list[Finding]:
        findings: list[Finding] = []
        run_start: int | None = None
        for i, v in enumerate(arr):
            if v < self.immobility_threshold:
                run_start = i if run_start is None else run_start
                continue
            if run_start is not None:
                findings.extend(self._emit_immobility(timestamps, run_start, i - 1))
                run_start = None
        if run_start is not None:
            findings.extend(self._emit_immobility(timestamps, run_start, len(arr) - 1))
        return findings

    def _emit_immobility(
        self, timestamps: list[datetime], start: int, end: int
    ) -> list[Finding]:
        length = end - start + 1
        if length < self.immobility_min_steps:
            return []
        severity = Severity.MEDIUM if length < self.immobility_min_steps * 2 else Severity.HIGH
        return [
            Finding(
                modality=Modality.MOVEMENT,
                severity=severity,
                description=(
                    f"Imobilidade prolongada por {length} intervalos consecutivos "
                    "(risco de úlcera de pressão/TVP)"
                ),
                score=min(length / (self.immobility_min_steps * 3), 1.0),
                timestamp=timestamps[start],
                metadata={"start_idx": start, "end_idx": end, "rule": "immobility"},
            )
        ]

    def _detect_spikes(
        self, timestamps: list[datetime], arr: np.ndarray
    ) -> list[Finding]:
        mean, std = float(arr.mean()), float(arr.std())
        if std < 1e-9:
            return []
        findings: list[Finding] = []
        for i, v in enumerate(arr):
            z = (v - mean) / std
            if z < self.spike_zscore:
                continue
            findings.append(
                Finding(
                    modality=Modality.MOVEMENT,
                    severity=Severity.HIGH,
                    description=(
                        f"Pico abrupto de movimentação (z={z:.1f}) — "
                        "possível queda ou agitação/crise"
                    ),
                    score=min(z / 6.0, 1.0),
                    timestamp=timestamps[i],
                    metadata={"idx": i, "zscore": z, "rule": "spike"},
                )
            )
        return findings
