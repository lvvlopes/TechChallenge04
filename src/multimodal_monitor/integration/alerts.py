"""Despacho de alertas para a equipe médica.

Canais suportados (via ``ALERT_CHANNEL``):
- ``console``: imprime o alerta formatado (padrão);
- ``file``:    anexa o alerta (JSON por linha) em ``outputs/alerts.jsonl``;
- ``webhook``: envia um POST JSON para ``ALERT_WEBHOOK_URL`` (Teams/Slack/etc.).
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from ..config import Settings, get_settings
from ..logging_config import get_logger
from ..schemas import Alert, Severity

logger = get_logger(__name__)
# safe_box + emoji desligado evita erros de encoding em consoles legados (cp1252)
console = Console(emoji=False)

_SEVERITY_STYLE = {
    Severity.LOW: "cyan",
    Severity.MEDIUM: "yellow",
    Severity.HIGH: "orange3",
    Severity.CRITICAL: "bold red",
}


class AlertManager:
    """Gerencia o despacho de alertas pelo canal configurado."""

    def __init__(
        self, settings: Settings | None = None, output_dir: str | Path = "outputs"
    ) -> None:
        self.settings = settings or get_settings()
        self.output_dir = Path(output_dir)

    def dispatch(self, alert: Alert) -> None:
        """Envia o alerta pelo canal configurado."""
        channel = self.settings.alert_channel
        if channel == "file":
            self._to_file(alert)
        elif channel == "webhook":
            self._to_webhook(alert)
        else:
            self._to_console(alert)

    # ------------------------------------------------------------------ #
    def _to_console(self, alert: Alert) -> None:
        style = _SEVERITY_STYLE.get(alert.severity, "white")
        console.print(
            Panel(
                alert.to_human(),
                title=f"[!] ALERTA CLINICO — {alert.severity.value.upper()}",
                border_style=style,
                expand=False,
            )
        )

    def _to_file(self, alert: Alert) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "alerts.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(alert.model_dump_json() + "\n")
        logger.info("Alerta persistido em %s", path)

    def _to_webhook(self, alert: Alert) -> None:
        url = self.settings.alert_webhook_url
        if not url:
            logger.warning("ALERT_CHANNEL=webhook mas ALERT_WEBHOOK_URL vazio; fallback console.")
            self._to_console(alert)
            return
        try:
            import httpx

            payload = {"text": alert.to_human(), **json.loads(alert.model_dump_json())}
            resp = httpx.post(url, json=payload, timeout=10.0)
            resp.raise_for_status()
            logger.info("Alerta enviado ao webhook (status %s).", resp.status_code)
        except Exception as exc:  # pragma: no cover - depende de rede
            logger.error("Falha ao enviar webhook (%s); fallback console.", exc)
            self._to_console(alert)
