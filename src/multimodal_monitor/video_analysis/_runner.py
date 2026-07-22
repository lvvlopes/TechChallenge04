"""Runner de análise de vídeo isolado em subprocesso.

Executado como ``python -m multimodal_monitor.video_analysis._runner <video>``.
Roda o :class:`VideoAnalysisPipeline` num processo Python limpo e imprime o
:class:`ModalityResult` serializado em JSON, prefixado pelo marcador
``__RESULT__`` na saída padrão.

Motivação: MediaPipe + ultralytics/torch + protobuf, carregados no mesmo
processo do servidor web (uvicorn, com endpoints sync em threadpool), colidem
de forma dependente da ordem de import (ex.: ``MessageFactory.GetPrototype``).
Isolar a análise pesada num subprocesso garante o mesmo comportamento da
execução direta (que funciona) e mantém o worker da API livre.
"""

from __future__ import annotations

import json
import sys

RESULT_MARKER = "__RESULT__"


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print(f"{RESULT_MARKER}{json.dumps({'error': 'missing video path'})}")
        return 2

    from .video_pipeline import VideoAnalysisPipeline

    result = VideoAnalysisPipeline().process(argv[0])
    # ModalityResult é um BaseModel (pydantic) — serializa direto.
    sys.stdout.write(RESULT_MARKER + result.model_dump_json() + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
