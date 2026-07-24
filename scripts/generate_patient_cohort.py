"""Gera uma coorte sintética de pacientes em um **único arquivo** JSON.

Cria N pacientes (padrão 20) com cenários "estável" e "crítico", cada um com
dados coerentes e reprodutíveis (seed derivada do índice) para as quatro
modalidades, e grava tudo em ``data/patients.json``:

- metadados (nome, idade, leito, cenário, queixa);
- sinais vitais e sinais de pose em formato **colunar** (compacto);
- prescrições;
- transcrição da consulta (fonte do texto quando não há Azure).

Binários (áudio ``.wav`` e vídeo ``.mp4``) não cabem em JSON e ficam numa pasta
plana ``data/patients_media/`` como ``<ID>.wav`` / ``<ID>.mp4``.

Uso:
    python scripts/generate_patient_cohort.py [--n 20] [--seed 100]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multimodal_monitor.synthetic import (  # noqa: E402
    CRITICAL_TRANSCRIPTS,
    STABLE_TRANSCRIPTS,
    generate_pose_frames,
    generate_prescriptions,
    generate_vitals,
)

OUT_FILE = Path("data/patients.json")
MEDIA_DIR = Path("data/patients_media")

# Nomes fictícios para a coorte (dados 100% sintéticos, não são pessoas reais).
NAMES = [
    "Ana Ribeiro", "Bruno Carvalho", "Carla Menezes", "Diego Fontes",
    "Eduarda Lima", "Felipe Andrade", "Gabriela Souza", "Henrique Dias",
    "Isabela Nunes", "João Pereira", "Karina Barros", "Lucas Almeida",
    "Marina Costa", "Nelson Tavares", "Olívia Rocha", "Paulo Cardoso",
    "Renata Farias", "Sérgio Moraes", "Tatiane Vieira", "Vitor Campos",
]

SECTORS = ["UTI", "Enfermaria", "Semi-intensiva", "Pós-operatório"]

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

# Arredondamento das séries (mantém o JSON compacto e legível).
ROUND = {
    "heart_rate": 1, "systolic_bp": 1, "diastolic_bp": 1,
    "spo2": 1, "respiratory_rate": 1, "temperature": 2,
}


def _scenario_for(index: int) -> str:
    """Alterna cenários de forma determinística (~metade de cada)."""
    return "critico" if index % 2 == 0 else "estavel"


def build_patient(index: int, seed_base: int) -> dict:
    pid = f"PAC-{index + 1:03d}"
    scenario = _scenario_for(index)
    critical = scenario == "critico"
    seed = seed_base + index

    # --- Sinais vitais (colunar) ---
    vdf = generate_vitals(seed=seed, inject_anomalies=critical)
    vitals = {"timestamp": [ts.isoformat() for ts in vdf["timestamp"]]}
    for col, nd in ROUND.items():
        vitals[col] = [round(float(v), nd) for v in vdf[col]]

    # --- Prescrições ---
    prescriptions = [
        {
            "timestamp": p.timestamp.isoformat(),
            "drug": p.drug,
            "dose_mg": p.dose_mg,
            "action": p.action,
        }
        for p in generate_prescriptions(seed=seed, inject_anomalies=critical)
    ]

    # --- Pose (colunar) ---
    poses = generate_pose_frames(seed=seed, inject_anomalies=critical)
    pose_frames = {
        "frame_index": [p.frame_index for p in poses],
        "timestamp_s": [round(p.timestamp_s, 3) for p in poses],
        "movement_index": [round(p.movement_index, 4) for p in poses],
        "trunk_angle": [
            None if p.trunk_angle is None else round(p.trunk_angle, 2) for p in poses
        ],
    }

    pool = CRITICAL_TRANSCRIPTS if critical else STABLE_TRANSCRIPTS
    complaint_pool = COMPLAINTS[scenario]

    return {
        "id": pid,
        "name": NAMES[index % len(NAMES)],
        "age": 40 + (index * 7) % 45,
        "bed": f"{SECTORS[index % len(SECTORS)]} · Leito {index % 12 + 1:02d}",
        "scenario": scenario,
        "chief_complaint": complaint_pool[index % len(complaint_pool)],
        "transcript": pool[index % len(pool)],
        "vitals": vitals,
        "prescriptions": prescriptions,
        "pose_frames": pose_frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera a coorte de pacientes.")
    parser.add_argument("--n", type=int, default=20, help="Número de pacientes")
    parser.add_argument("--seed", type=int, default=100, help="Seed base")
    args = parser.parse_args()

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)

    patients = [build_patient(i, args.seed) for i in range(args.n)]
    manifest = {
        "description": "Coorte sintética de pacientes — FIAP TC Fase 4",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "count": len(patients),
        "media_dir": MEDIA_DIR.name,
        "patients": patients,
    }
    OUT_FILE.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    n_crit = sum(1 for p in patients if p["scenario"] == "critico")
    size_kb = OUT_FILE.stat().st_size / 1024
    print(f"Coorte gerada: {OUT_FILE.resolve()} ({size_kb:.0f} KB)")
    print(f"  {len(patients)} pacientes ({n_crit} críticos, {len(patients) - n_crit} estáveis)")
    print(f"  binários (áudio/vídeo) em: {MEDIA_DIR.resolve()}")


if __name__ == "__main__":
    main()
