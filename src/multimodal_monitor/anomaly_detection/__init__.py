"""Detecção de anomalias (RF03).

Submódulos:
    - ``vitals``:        séries temporais de sinais vitais.
    - ``prescriptions``: alterações inesperadas na evolução de prescrições.
    - ``movement``:      padrões de movimentação do paciente.
"""

from __future__ import annotations

from .movement import MovementAnomalyDetector
from .prescriptions import PrescriptionAnomalyDetector
from .vitals import VitalsAnomalyDetector

__all__ = [
    "VitalsAnomalyDetector",
    "PrescriptionAnomalyDetector",
    "MovementAnomalyDetector",
]
