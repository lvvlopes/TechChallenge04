"""Coorte de pacientes persistida em um único arquivo JSON.

Carrega ``data/patients.json`` — que contém **todos** os pacientes com seus
metadados, sinais vitais, prescrições, sinais de pose e transcrição — e monta um
:class:`MonitoringInput` a partir do ID do paciente, "amarrando" automaticamente
as quatro modalidades ao mesmo paciente.

Layout::

    data/
    ├── patients.json       # tudo, exceto binários
    └── patients_media/     # binários (não cabem em JSON)
        ├── PAC-001.wav     # áudio da consulta (TTS pt-BR)
        └── PAC-001.mp4     # vídeo real (opcional, opt-in por paciente)

As séries longas (vitais e pose) são gravadas em formato **colunar**
(``{"heart_rate": [...], "spo2": [...]}``) — bem mais compacto e legível que
uma lista de objetos por linha.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .anomaly_detection.prescriptions import PrescriptionEvent
from .integration.orchestrator import MonitoringInput
from .video_analysis.pose_analyzer import PoseFrame

DEFAULT_COHORT_FILE = Path("data/patients.json")
MEDIA_DIRNAME = "patients_media"

VITALS_COLUMNS = (
    "timestamp",
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "spo2",
    "respiratory_rate",
    "temperature",
)


@dataclass
class PatientRecord:
    """Metadados de um paciente da coorte (não inclui as séries de dados)."""

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
    """Acesso de leitura à coorte de pacientes (arquivo único JSON)."""

    path: Path = field(default_factory=lambda: DEFAULT_COHORT_FILE)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self._cache: dict | None = None

    # ------------------------------------------------------------------ #
    @property
    def available(self) -> bool:
        return self.path.exists()

    @property
    def media_dir(self) -> Path:
        """Pasta dos binários, irmã do arquivo JSON."""
        return self.path.parent / MEDIA_DIRNAME

    def _load(self) -> dict:
        if self._cache is None:
            self._cache = json.loads(self.path.read_text(encoding="utf-8"))
        return self._cache

    def _patient_dict(self, patient_id: str) -> dict | None:
        for p in self._load().get("patients", []):
            if p.get("id") == patient_id:
                return p
        return None

    # ------------------------------------------------------------------ #
    def list_patients(self) -> list[PatientRecord]:
        """Retorna os metadados de todos os pacientes da coorte."""
        if not self.available:
            return []
        known = {"id", "name", "age", "bed", "scenario", "chief_complaint"}
        return [
            PatientRecord(**{k: v for k, v in p.items() if k in known})
            for p in self._load().get("patients", [])
        ]

    def get(self, patient_id: str) -> PatientRecord | None:
        p = self._patient_dict(patient_id) if self.available else None
        if p is None:
            return None
        known = {"id", "name", "age", "bed", "scenario", "chief_complaint"}
        return PatientRecord(**{k: v for k, v in p.items() if k in known})

    def audio_file(self, patient_id: str) -> Path | None:
        """Caminho do áudio da consulta do paciente, se existir."""
        f = self.media_dir / f"{patient_id}.wav"
        return f if f.exists() else None

    def video_file(self, patient_id: str) -> Path | None:
        """Caminho do vídeo real do paciente, se existir.

        Convenção: ``data/patients_media/<ID>.mp4``. Quando presente, os sinais
        de pose do paciente devem ser extraídos dele por
        ``scripts/extract_patient_pose.py``.
        """
        f = self.media_dir / f"{patient_id}.mp4"
        return f if f.exists() else None

    # ------------------------------------------------------------------ #
    def load_input(self, patient_id: str) -> MonitoringInput:
        """Monta o :class:`MonitoringInput` de um paciente a partir do JSON."""
        p = self._patient_dict(patient_id) if self.available else None
        if p is None:
            raise KeyError(f"Paciente não encontrado na coorte: {patient_id}")

        # --- Sinais vitais (formato colunar -> DataFrame) ---
        vitals = None
        vcols = p.get("vitals") or {}
        if vcols:
            vitals = pd.DataFrame({c: vcols[c] for c in VITALS_COLUMNS if c in vcols})
            if "timestamp" in vitals.columns:
                vitals["timestamp"] = pd.to_datetime(vitals["timestamp"])

        # --- Prescrições ---
        prescriptions = [
            PrescriptionEvent(
                timestamp=pd.to_datetime(e["timestamp"]),
                drug=str(e["drug"]),
                dose_mg=float(e["dose_mg"]),
                action=str(e.get("action", "prescribe")),
            )
            for e in p.get("prescriptions", [])
        ]

        # --- Áudio (o STT em modo mock usa a transcrição do JSON) ---
        audio_path = self.audio_file(patient_id)

        # --- Pose (formato colunar). Para pacientes com vídeo real, a pose é
        # extraída do arquivo por scripts/extract_patient_pose.py. ---
        pose_frames: list[PoseFrame] | None = None
        pcols = p.get("pose_frames") or {}
        if pcols and pcols.get("frame_index"):
            angles = pcols.get("trunk_angle") or []
            pose_frames = [
                PoseFrame(
                    frame_index=int(pcols["frame_index"][i]),
                    timestamp_s=float(pcols["timestamp_s"][i]),
                    movement_index=float(pcols["movement_index"][i]),
                    trunk_angle=(
                        None
                        if i >= len(angles) or angles[i] is None
                        else float(angles[i])
                    ),
                )
                for i in range(len(pcols["frame_index"]))
            ]

        return MonitoringInput(
            patient_id=patient_id,
            vitals=vitals,
            audio_path=audio_path,
            pose_frames=pose_frames,
            prescriptions=prescriptions,
            # fonte única do texto: se o Azure transcrever o .wav, esse texto
            # prevalece; senão, usa-se a transcrição gravada no dataset.
            transcript=p.get("transcript") or None,
        )

    # ------------------------------------------------------------------ #
    def transcript(self, patient_id: str) -> str:
        """Transcrição da consulta do paciente (texto no JSON)."""
        p = self._patient_dict(patient_id) if self.available else None
        return (p or {}).get("transcript", "")
