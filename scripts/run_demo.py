"""Executa o cenário de demonstração multimodal.

Uso:
    python scripts/run_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# permite executar sem instalar o pacote (adiciona ./src ao path)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multimodal_monitor.demo import run_demo  # noqa: E402


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
