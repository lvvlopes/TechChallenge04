"""Extrai a pose real do vídeo de cada paciente para ``pose_frames.csv``.

Para cada pasta de paciente que contenha ``video_teste.mp4``, roda a análise
postural (MediaPipe Pose) sobre o vídeo real e sobrescreve o ``pose_frames.csv``
com os sinais extraídos. Assim, a análise da coorte usa a pose do vídeo de
verdade — mas em memória (rápido e confiável), sem executar a análise pesada
dentro do servidor web.

Rode este script sempre que adicionar/trocar o ``video_teste.mp4`` de um
paciente.

Uso:
    python scripts/extract_patient_pose.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multimodal_monitor.video_analysis.pose_analyzer import PoseAnalyzer  # noqa: E402

PATIENTS = Path("data/patients")
VIDEO_NAME = "video_teste.mp4"


def main() -> None:
    if not PATIENTS.exists():
        print("Coorte não encontrada. Rode antes: python scripts/generate_patient_cohort.py")
        return

    analyzer = PoseAnalyzer()
    if not analyzer.available:
        print(
            "MediaPipe/OpenCV indisponíveis. Instale a extra de visão: "
            'pip install ".[vision]"'
        )
        return

    found = 0
    for pdir in sorted(PATIENTS.glob("PAC-*")):
        video = pdir / VIDEO_NAME
        if not video.exists():
            continue
        found += 1
        print(f"  {pdir.name}: extraindo pose de {VIDEO_NAME}…", flush=True)
        frames = analyzer.analyze_video(video)
        pd.DataFrame([f.__dict__ for f in frames]).to_csv(pdir / "pose_frames.csv", index=False)
        print(f"    {len(frames)} quadros -> pose_frames.csv")

    if found == 0:
        print(
            "Nenhum paciente com video_teste.mp4 na pasta. Para ativar o vídeo "
            "real de um paciente, copie data/samples/video_teste.mp4 para a "
            "pasta dele (ex.: data/patients/PAC-001/)."
        )
    else:
        print(f"Concluído: pose real extraída para {found} paciente(s).")


if __name__ == "__main__":
    main()
