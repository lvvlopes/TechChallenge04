"""Cenário de demonstração ponta-a-ponta do monitor multimodal.

Gera dados sintéticos com anomalias plantadas para as quatro fontes (sinais
vitais, prescrições, pose de vídeo e áudio de consulta), executa o pipeline
multimodal completo e imprime um resumo com o alerta clínico resultante.

Executável via ``mmonitor demo`` ou ``python -m multimodal_monitor.demo``.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from .integration.orchestrator import MonitoringInput, PatientMonitor
from .logging_config import setup_logging
from .synthetic import (
    SAMPLE_TRANSCRIPTS,
    generate_pose_frames,
    generate_prescriptions,
    generate_vitals,
)

console = Console()

DATA_DIR = Path("data/samples")


def _ensure_sample_transcript() -> Path:
    """Escreve o transcript de referência (usado pelo STT em modo mock)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    audio_stub = DATA_DIR / "consulta_critica.wav"
    transcript = audio_stub.with_suffix(".txt")
    if not transcript.exists():
        transcript.write_text(SAMPLE_TRANSCRIPTS["critico"], encoding="utf-8")
    return audio_stub


def _print_modalities(report) -> None:
    table = Table(title="Resultado por modalidade", show_lines=False)
    table.add_column("Modalidade", style="cyan")
    table.add_column("Risco", justify="right")
    table.add_column("Achados", justify="right")
    table.add_column("Resumo", overflow="fold")
    for r in report.modality_results:
        table.add_row(
            r.modality.value,
            f"{r.risk_score:.2f}",
            str(len(r.findings)),
            r.summary,
        )
    console.print(table)


def run_demo(patient_id: str = "PAC-001") -> None:
    setup_logging()
    console.rule("[bold]Monitoramento Multimodal de Pacientes — Demonstração")

    audio_stub = _ensure_sample_transcript()

    data = MonitoringInput(
        patient_id=patient_id,
        vitals=generate_vitals(inject_anomalies=True),
        prescriptions=generate_prescriptions(inject_anomalies=True),
        pose_frames=generate_pose_frames(inject_anomalies=True),
        audio_path=audio_stub,
    )

    monitor = PatientMonitor(dispatch_alerts=True)
    report = monitor.run(data)

    console.print()
    _print_modalities(report)
    console.print()
    console.print(f"[bold]Score de risco multimodal:[/bold] {report.risk_score:.2f}")

    if report.alert is None:
        console.print("[green]Nenhum alerta disparado (risco abaixo do limiar).[/green]")
    else:
        console.rule("[bold red]Alerta despachado à equipe médica")

    console.rule("[bold]Fim da demonstração")


if __name__ == "__main__":
    run_demo()
