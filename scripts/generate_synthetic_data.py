"""Gera e persiste datasets sintéticos em ``data/samples/``.

Uso:
    python scripts/generate_synthetic_data.py [--seed 42]

Arquivos gerados:
    data/samples/vitals.csv            — série de sinais vitais
    data/samples/prescriptions.csv     — evolução de prescrições
    data/samples/pose_frames.csv       — sinais posturais de vídeo
    data/samples/consulta_critica.txt  — transcrição de referência (STT mock)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multimodal_monitor.synthetic import (  # noqa: E402
    SAMPLE_TRANSCRIPTS,
    generate_pose_frames,
    generate_prescriptions,
    generate_vitals,
)

OUT = Path("data/samples")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera datasets sintéticos.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true", help="Sem anomalias plantadas.")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    inject = not args.clean

    vitals = generate_vitals(seed=args.seed, inject_anomalies=inject)
    vitals.to_csv(OUT / "vitals.csv", index=False)

    presc = generate_prescriptions(seed=args.seed, inject_anomalies=inject)
    pd.DataFrame([p.model_dump() for p in presc]).to_csv(OUT / "prescriptions.csv", index=False)

    poses = generate_pose_frames(seed=args.seed, inject_anomalies=inject)
    pd.DataFrame([p.__dict__ for p in poses]).to_csv(OUT / "pose_frames.csv", index=False)

    (OUT / "consulta_critica.txt").write_text(SAMPLE_TRANSCRIPTS["critico"], encoding="utf-8")
    (OUT / "consulta_estavel.txt").write_text(SAMPLE_TRANSCRIPTS["estavel"], encoding="utf-8")

    print(f"Datasets sintéticos gerados em {OUT.resolve()}")


if __name__ == "__main__":
    main()
