"""Extrai a pose real do vídeo de cada paciente para dentro de ``data/patients.json``.

Para cada paciente que tenha um vídeo em ``data/patients_media/<ID>.mp4``, roda a
análise postural (MediaPipe Pose) sobre o vídeo real e **substitui** os sinais de
pose do paciente no arquivo da coorte. Assim, a análise usa a pose do vídeo de
verdade — mas em memória (rápido e confiável), sem executar a análise pesada
dentro do servidor web.

Rode este script sempre que adicionar/trocar o vídeo de um paciente.

Uso:
    python scripts/extract_patient_pose.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multimodal_monitor.video_analysis.pose_analyzer import PoseAnalyzer  # noqa: E402

COHORT_FILE = Path("data/patients.json")
MEDIA_DIR = Path("data/patients_media")


def main() -> None:
    if not COHORT_FILE.exists():
        print("Coorte não encontrada. Rode antes: python scripts/generate_patient_cohort.py")
        return

    analyzer = PoseAnalyzer()
    if not analyzer.available:
        print(
            "MediaPipe/OpenCV indisponíveis. Instale a extra de visão: "
            'pip install ".[vision]"'
        )
        return

    cohort = json.loads(COHORT_FILE.read_text(encoding="utf-8"))
    found = 0

    for patient in cohort.get("patients", []):
        video = MEDIA_DIR / f"{patient['id']}.mp4"
        if not video.exists():
            continue
        found += 1
        print(f"  {patient['id']}: extraindo pose de {video.name}…", flush=True)
        frames = analyzer.analyze_video(video)
        patient["pose_frames"] = {
            "frame_index": [f.frame_index for f in frames],
            "timestamp_s": [round(f.timestamp_s, 3) for f in frames],
            "movement_index": [round(f.movement_index, 4) for f in frames],
            "trunk_angle": [
                None if f.trunk_angle is None else round(f.trunk_angle, 2) for f in frames
            ],
        }
        patient["pose_source"] = "video"
        print(f"    {len(frames)} quadros -> pose real gravada na coorte")

    if found == 0:
        print(
            f"Nenhum paciente com vídeo em {MEDIA_DIR}/. Para ativar o vídeo real "
            "de um paciente, copie um .mp4 para lá com o nome do ID "
            "(ex.: data/patients_media/PAC-001.mp4)."
        )
        return

    COHORT_FILE.write_text(
        json.dumps(cohort, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"Concluído: pose real extraída para {found} paciente(s).")


if __name__ == "__main__":
    main()
