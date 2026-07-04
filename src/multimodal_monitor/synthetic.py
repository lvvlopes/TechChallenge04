"""Geração de dados sintéticos para demonstração reprodutível.

Produz cenários controlados (com anomalias plantadas) para cada modalidade,
permitindo demonstrar o pipeline sem depender de datasets externos ou de
credenciais de nuvem. Todos os geradores aceitam ``seed`` para reprodutibilidade.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .anomaly_detection.prescriptions import PrescriptionEvent
from .video_analysis.pose_analyzer import PoseFrame


def generate_vitals(
    n: int = 240,
    seed: int = 42,
    inject_anomalies: bool = True,
    interval_seconds: int = 60,
) -> pd.DataFrame:
    """Gera uma série temporal de sinais vitais com anomalias plantadas.

    A linha de base é fisiologicamente plausível; se ``inject_anomalies`` for
    True, insere uma dessaturação de SpO2, um pico de FC e um evento de
    hipertensão em janelas específicas.
    """
    rng = np.random.default_rng(seed)
    t0 = datetime.now(timezone.utc) - timedelta(seconds=n * interval_seconds)
    timestamps = [t0 + timedelta(seconds=i * interval_seconds) for i in range(n)]

    # Ruído calibrado para variabilidade fisiológica típica de leito de UTI
    hr = rng.normal(75, 2.5, n)
    sbp = rng.normal(120, 4, n)
    dbp = rng.normal(78, 3, n)
    spo2 = rng.normal(97, 0.5, n)
    rr = rng.normal(16, 1.0, n)
    temp = rng.normal(36.6, 0.15, n)

    if inject_anomalies and n >= 200:
        # Evento 1: dessaturação progressiva de SpO2 (crise respiratória)
        spo2[120:135] = np.linspace(96, 85, 15)
        rr[120:135] = np.linspace(17, 27, 15)
        # Evento 2: taquicardia abrupta
        hr[160:168] = np.linspace(80, 135, 8)
        # Evento 3: pico hipertensivo
        sbp[190:200] = np.linspace(125, 185, 10)
        dbp[190:200] = np.linspace(80, 118, 10)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "heart_rate": hr.round(1),
            "systolic_bp": sbp.round(1),
            "diastolic_bp": dbp.round(1),
            "spo2": spo2.round(1),
            "respiratory_rate": rr.round(1),
            "temperature": temp.round(2),
        }
    )


def generate_prescriptions(
    seed: int = 42, inject_anomalies: bool = True
) -> list[PrescriptionEvent]:
    """Gera uma evolução de prescrições com anomalias plantadas."""
    t0 = datetime.now(timezone.utc) - timedelta(hours=12)
    # base sem interações conhecidas entre si
    events = [
        PrescriptionEvent(timestamp=t0, drug="paracetamol", dose_mg=750),
        PrescriptionEvent(timestamp=t0 + timedelta(hours=2), drug="amoxicilina", dose_mg=500),
        PrescriptionEvent(timestamp=t0 + timedelta(hours=4), drug="dipirona", dose_mg=1000),
    ]
    if inject_anomalies:
        # salto abrupto de dose de dipirona (1000 -> 3000, 3x)
        events.append(
            PrescriptionEvent(timestamp=t0 + timedelta(hours=6), drug="dipirona", dose_mg=3000)
        )
        # introduz warfarina e depois ibuprofeno → interação conhecida
        events.append(
            PrescriptionEvent(
                timestamp=t0 + timedelta(hours=6, minutes=30),
                drug="warfarina",
                dose_mg=5,
            )
        )
        events.append(
            PrescriptionEvent(
                timestamp=t0 + timedelta(hours=7), drug="ibuprofeno", dose_mg=600
            )
        )
    return events


def generate_pose_frames(
    n: int = 120, seed: int = 42, inject_anomalies: bool = True, fps: float = 5.0
) -> list[PoseFrame]:
    """Gera sinais posturais sintéticos (movimento + ângulo de tronco)."""
    rng = np.random.default_rng(seed)
    movement = np.abs(rng.normal(0.25, 0.08, n))
    trunk = rng.normal(10, 3, n)

    if inject_anomalies and n >= 100:
        # imobilidade prolongada (movimento ~0)
        movement[30:50] = rng.normal(0.02, 0.005, 20)
        # pico de movimento (possível queda)
        movement[80] = 1.0
        # desvio postural acentuado
        trunk[60:70] = np.linspace(20, 60, 10)

    return [
        PoseFrame(
            frame_index=i,
            timestamp_s=i / fps,
            movement_index=float(np.clip(movement[i], 0, 1)),
            trunk_angle=float(trunk[i]),
        )
        for i in range(n)
    ]


SAMPLE_TRANSCRIPTS = {
    "critico": (
        "Doutor, estou sentindo muita falta de ar desde ontem e uma dor no peito "
        "que não passa. Fico muito cansado com qualquer esforço e tive uma tontura "
        "forte hoje de manhã. Estou preocupado, me sinto muito mal."
    ),
    "estavel": (
        "Estou me sentindo bem melhor hoje, doutor. A dor diminuiu bastante e "
        "consegui dormir tranquilo. Estou mais disposto e animado com a recuperação."
    ),
}
