"""Interface de linha de comando do monitor multimodal.

Uso:
    mmonitor demo                 # executa o cenário de demonstração completo
    mmonitor api                  # sobe a API (uvicorn)
    mmonitor version
"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .logging_config import setup_logging


def _cmd_demo(_args: argparse.Namespace) -> int:
    from .demo import run_demo

    run_demo()
    return 0


def _cmd_api(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "multimodal_monitor.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def _cmd_version(_args: argparse.Namespace) -> int:
    print(f"multimodal-monitor {__version__}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmonitor",
        description="Monitoramento Multimodal de Pacientes — FIAP TC Fase 4.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="Executa o cenário de demonstração.")
    p_demo.set_defaults(func=_cmd_demo)

    p_api = sub.add_parser("api", help="Sobe a API REST (uvicorn).")
    p_api.add_argument("--host", default="127.0.0.1")
    p_api.add_argument("--port", type=int, default=8000)
    p_api.add_argument("--reload", action="store_true")
    p_api.set_defaults(func=_cmd_api)

    p_ver = sub.add_parser("version", help="Mostra a versão.")
    p_ver.set_defaults(func=_cmd_version)

    return parser


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
