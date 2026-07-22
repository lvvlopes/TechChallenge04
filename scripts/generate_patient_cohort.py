"""Gera uma coorte sintética de pacientes persistida em ``data/patients/``.

Cria N pacientes (padrão 20) com cenários "estável" e "crítico", cada um com
dados coerentes e reprodutíveis (seed derivada do ID) para as quatro
modalidades: sinais vitais, prescrições, transcrição de consulta e sinais de
pose. Também grava o manifesto ``cohort.json`` com os metadados de cada
paciente.

Uso:
    python scripts/generate_patient_cohort.py [--n 20] [--seed 100]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multimodal_monitor.synthetic import (  # noqa: E402
    CRITICAL_TRANSCRIPTS,
    STABLE_TRANSCRIPTS,
    generate_pose_frames,
    generate_prescriptions,
    generate_vitals,
)

OUT = Path("data/patients")
REAL_VIDEO = "data/samples/video_teste.mp4"

# Nomes fictícios para a coorte (dados 100% sintéticos, não são pessoas reais).
NAMES = [
    "Ana Ribeiro", "Bruno Carvalho", "Carla Menezes", "Diego Fontes",
    "Eduarda Lima", "Felipe Andrade", "Gabriela Souza", "Henrique Dias",
    "Isabela Nunes", "João Pereira", "Karina Barros", "Lucas Almeida",
    "Marina Costa", "Nelson Tavares", "Olívia Rocha", "Paulo Cardoso",
    "Renata Farias", "Sérgio Moraes", "Tatiane Vieira", "Vitor Campos",
]

SECTORS = ["UTI", "Enfermaria", "Semi-intensiva", "Pós-operatório"]

# Queixa principal por cenário (curta, para exibição na lista).
COMPLAINTS = {
    "critico": [
        "Dispneia e dor torácica",
        "Queda com rebaixamento",
        "Taquicardia e fadiga intensa",
        "Dessaturação progressiva",
    ],
    "estavel": [
        "Acompanhamento pós-operatório",
        "Reavaliação de rotina",
        "Evolução favorável",
        "Fisioterapia em andamento",
    ],
}


def _scenario_for(index: int) -> str:
    """Alterna cenários de forma determinística (~metade de cada)."""
    return "critico" if index % 2 == 0 else "estavel"


def build_patient(index: int, seed_base: int) -> dict:
    pid = f"PAC-{index + 1:03d}"
    scenario = _scenario_for(index)
    critical = scenario == "critico"
    seed = seed_base + index
    pdir = OUT / pid
    pdir.mkdir(parents=True, exist_ok=True)

    # --- Sinais vitais (persistidos, coerentes com o cenário) ---
    vitals = generate_vitals(seed=seed, inject_anomalies=critical)
    vitals.to_csv(pdir / "vitals.csv", index=False)

    # --- Prescrições ---
    presc = generate_prescriptions(seed=seed, inject_anomalies=critical)
    pd.DataFrame([p.model_dump() for p in presc]).to_csv(
        pdir / "prescriptions.csv", index=False
    )

    # --- Pose (vídeo pré-computado) ---
    poses = generate_pose_frames(seed=seed, inject_anomalies=critical)
    pd.DataFrame([p.__dict__ for p in poses]).to_csv(pdir / "pose_frames.csv", index=False)

    # --- Transcrição da consulta (fonte do STT em modo mock) ---
    pool = CRITICAL_TRANSCRIPTS if critical else STABLE_TRANSCRIPTS
    transcript = pool[index % len(pool)]
    (pdir / "consulta.txt").write_text(transcript, encoding="utf-8")

    # --- Metadados do manifesto ---
    complaint_pool = COMPLAINTS[scenario]
    return {
        "id": pid,
        "name": NAMES[index % len(NAMES)],
        "age": 40 + (index * 7) % 45,
        "bed": f"{SECTORS[index % len(SECTORS)]} · Leito {index % 12 + 1:02d}",
        "scenario": scenario,
        "chief_complaint": complaint_pool[index % len(complaint_pool)],
        # vídeo real vinculado aos críticos (o vídeo de teste exibe postura
        # horizontal = compatível com queda); estáveis usam apenas a pose.
        "video": REAL_VIDEO if critical else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera a coorte de pacientes.")
    parser.add_argument("--n", type=int, default=20, help="Número de pacientes")
    parser.add_argument("--seed", type=int, default=100, help="Seed base")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    patients = [build_patient(i, args.seed) for i in range(args.n)]

    manifest = {
        "description": "Coorte sintética de pacientes — FIAP TC Fase 4",
        "count": len(patients),
        "patients": patients,
    }
    (OUT / "cohort.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    n_crit = sum(1 for p in patients if p["scenario"] == "critico")
    print(f"Coorte gerada em {OUT.resolve()}")
    print(f"  {len(patients)} pacientes ({n_crit} críticos, {len(patients) - n_crit} estáveis)")


if __name__ == "__main__":
    main()
