"""Coorte de pacientes persistida em disco.

Carrega uma coorte de pacientes (manifesto + dados por paciente) gravada em
``data/patients/`` e monta um :class:`MonitoringInput` a partir do ID do
paciente, "amarrando" automaticamente as quatro modalidades (sinais vitais,
prescrições, áudio/transcrição e vídeo/pose) ao mesmo paciente.

Layout esperado::

    data/patients/
        cohort.json                 # manifesto: lista de pacientes + metadados
        PAC-001/
            vitals.csv              # série temporal de sinais vitais
            prescriptions.csv       # evolução de prescrições
            consulta.wav            # (opcional) áudio real; senão usa .txt
            consulta.txt            # transcrição de referência (STT em mock)
            pose_frames.csv         # sinais de pose pré-computados
        PAC-002/ ...
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

import pandas as pd

from .anomaly_detection.prescriptions import PrescriptionEvent
from .integration.orchestrator import MonitoringInput
from .video_analysis.pose_analyzer import PoseFrame

DEFAULT_COHORT_ROOT = Path("data/patients")
MANIFEST_NAME = "cohort.json"


@dataclass
class PatientRecord:
    """Metadados de um paciente da coorte (não inclui os dados brutos)."""

    id: str
    name: str
    age: int
    bed: str
    scenario: str  # "estavel" | "critico"
    chief_complaint: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "age": self.age,
            "bed": self.bed,
            "scenario": self.scenario,
            "chief_complaint": self.chief_complaint,
        }


@dataclass
class PatientCohort:
    """Acesso de leitura à coorte de pacientes persistida."""

    root: Path = field(default_factory=lambda: DEFAULT_COHORT_ROOT)

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    # ------------------------------------------------------------------ #
    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_NAME

    @property
    def available(self) -> bool:
        return self.manifest_path.exists()

    def list_patients(self) -> list[PatientRecord]:
        """Lê o manifesto e retorna os registros de todos os pacientes."""
        if not self.available:
            return []
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        known = {f.name for f in fields(PatientRecord)}
        return [
            PatientRecord(**{k: v for k, v in p.items() if k in known})
            for p in data.get("patients", [])
        ]

    def get(self, patient_id: str) -> PatientRecord | None:
        return next((p for p in self.list_patients() if p.id == patient_id), None)

    def video_file(self, patient_id: str) -> Path | None:
        """Retorna o caminho do vídeo real do paciente, se existir na pasta.

        Convenção: se houver um arquivo ``video_teste.mp4`` dentro da pasta do
        paciente, ele é usado para análise de vídeo real (MediaPipe + YOLOv8).
        """
        vf = self.root / patient_id / "video_teste.mp4"
        return vf if vf.exists() else None

    # ------------------------------------------------------------------ #
    def load_input(self, patient_id: str) -> MonitoringInput:
        """Monta o :class:`MonitoringInput` de um paciente a partir dos arquivos.

        Regra do vídeo: se a pasta do paciente contiver ``video_teste.mp4``,
        esse arquivo é processado como vídeo real (MediaPipe + YOLOv8);
        caso contrário, usa-se a pose pré-computada (``pose_frames.csv``).
        """
        record = self.get(patient_id)
        if record is None:
            raise KeyError(f"Paciente não encontrado na coorte: {patient_id}")

        pdir = self.root / patient_id

        # --- Sinais vitais ---
        vitals = None
        vpath = pdir / "vitals.csv"
        if vpath.exists():
            vitals = pd.read_csv(vpath, parse_dates=["timestamp"])

        # --- Prescrições ---
        prescriptions: list[PrescriptionEvent] = []
        ppath = pdir / "prescriptions.csv"
        if ppath.exists():
            pdf = pd.read_csv(ppath, parse_dates=["timestamp"])
            prescriptions = [
                PrescriptionEvent(
                    timestamp=row["timestamp"],
                    drug=str(row["drug"]),
                    dose_mg=float(row["dose_mg"]),
                    action=str(row.get("action", "prescribe")),
                )
                for _, row in pdf.iterrows()
            ]

        # --- Áudio (aponta para .wav; o STT em mock lê o .txt irmão) ---
        audio_path: str | Path | None = None
        if (pdir / "consulta.txt").exists():
            audio_path = pdir / "consulta.wav"

        # --- Vídeo / pose ---
        # Sempre usa pose pré-computada (em memória, rápido e confiável). Para
        # pacientes com 'video_teste.mp4', o pose_frames.csv é extraído do vídeo
        # real por scripts/extract_patient_pose.py — assim os achados vêm do
        # vídeo de verdade, sem rodar a análise pesada dentro do servidor web.
        pose_frames: list[PoseFrame] | None = None
        fpath = pdir / "pose_frames.csv"
        if fpath.exists():
            fdf = pd.read_csv(fpath)
            pose_frames = [
                PoseFrame(
                    frame_index=int(row["frame_index"]),
                    timestamp_s=float(row["timestamp_s"]),
                    movement_index=float(row["movement_index"]),
                    trunk_angle=(
                        None if pd.isna(row.get("trunk_angle")) else float(row["trunk_angle"])
                    ),
                )
                for _, row in fdf.iterrows()
            ]

        return MonitoringInput(
            patient_id=patient_id,
            vitals=vitals,
            audio_path=audio_path,
            pose_frames=pose_frames,
            prescriptions=prescriptions,
        )
