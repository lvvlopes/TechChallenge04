# Checklist de Requisitos — Tech Challenge Fase 4

**Curso:** FIAP Pós-Tech IADT (8IADT) · **Fase 4** · Monitoramento Multimodal de Pacientes
**Repositório:** *(inserir URL do GitHub)* · **Vídeo demonstrativo:** *(inserir URL YouTube/Vimeo)*

Este documento mapeia cada requisito do enunciado ("8IADT - Fase 4 - Tech
Challenge") para a implementação correspondente, com instruções de
verificação para a banca. Comandos assumem Python 3.10+ na raiz do projeto,
com `pip install -r requirements.txt` executado.

---

## 1. Análise de Vídeo (RF01)

| ✔ | Requisito do enunciado | Implementação | Como verificar |
|---|---|---|---|
| ✅ | Processar vídeos clínicos (fisioterapia / cirurgias gravadas) | `src/multimodal_monitor/video_analysis/video_pipeline.py` — `VideoAnalysisPipeline.process(video)` decodifica qualquer vídeo suportado pelo OpenCV | `python -c "from multimodal_monitor.video_analysis import VideoAnalysisPipeline; print(VideoAnalysisPipeline().process('data/samples/video_teste.mp4').summary)"` |
| ✅ | Detectar movimentos/eventos fora do padrão esperado | Índice de movimento por quadro + detector de imobilidade prolongada e picos abruptos (`anomaly_detection/movement.py`), reutilizado pelo pipeline de vídeo | Mesmo comando acima — achados listados no retorno |
| ✅* | Análise postural (enunciado sugere "modelos **como** OpenPose") | **MediaPipe Pose** (33 landmarks corporais; mesma finalidade do OpenPose, roda em CPU) — `video_analysis/pose_analyzer.py`. Extrai `movement_index` e `trunk_angle` por quadro; regra de desvio postural ≥ 35° | Vídeo com pessoa na horizontal gera achados "Desvio postural: inclinação de tronco de ~90°" |
| ✅ | YOLOv8 para detecção de objetos e áreas críticas | `video_analysis/object_detector.py` via `ultralytics` (YOLOv8n) | `python -c "from multimodal_monitor.video_analysis import ObjectDetector; [print(d.label, d.confidence) for d in ObjectDetector().detect_video('data/samples/video_teste.mp4', sample_rate=30)]"` |
| ✅ | Gerar relatórios automáticos indicando desvios/falhas | Cada análise produz `ModalityResult` com sumário + lista de achados tipados (`Finding`: severidade, descrição, score, timestamp, metadados); serializado em JSON pela API | `POST /intake` com upload de vídeo → campo `modalities.video` da resposta |

\* *Substituição justificada no relatório técnico (§3.1): OpenPose exige compilação C++/CUDA; MediaPipe Pose entrega os mesmos landmarks para análise postural com instalação `pip`. O contrato interno (`PoseFrame`) é agnóstico ao modelo.*

## 2. Análise de Áudio (RF02)

| ✔ | Requisito do enunciado | Implementação | Como verificar |
|---|---|---|---|
| ✅ | Processar áudios de consultas médicas | `audio_analysis/audio_pipeline.py` — áudio → transcrição → insights → achados | Tela `/intake`: gravar pelo microfone (WAV 16 kHz mono gerado no navegador) ou upload de arquivo |
| ✅ | Detectar alterações indicativas de condições médicas (cansaço, dificuldades respiratórias) | Dicionário clínico pt-BR com severidade por termo (`CRITICAL_TERMS` em `text_analytics.py`): "cansaço/fadiga" → MEDIUM, "falta de ar/dificuldade para respirar" → HIGH, "dor no peito" → CRITICAL, etc. | `python -c "from multimodal_monitor.audio_analysis import TextAnalytics; print(TextAnalytics().analyze('estou cansado e com falta de ar').critical_terms)"` |
| ✅ | **Azure Speech to Text** para transcrever | `audio_analysis/speech_to_text.py` — SDK oficial, reconhecimento contínuo em `pt-BR`; conversão automática de formatos não-WAV via ffmpeg; fallback offline (mock) sem credenciais | `GET /health` → `capabilities.azure_speech: true`; transcrição real retorna `source: "azure"` |
| ✅ | Termos críticos e sentimentos com **Azure Text Analytics** | `audio_analysis/text_analytics.py` — sentimento via Azure Language (`analyze_sentiment`, pt); termos críticos via dicionário local; fallback léxico offline | `GET /health` → `capabilities.azure_language: true`; painel de transcrição do dashboard realça termos por severidade |

## 3. Detecção de Anomalias (RF03)

| ✔ | Requisito do enunciado | Implementação | Como verificar |
|---|---|---|---|
| ✅ | Séries temporais de sinais vitais (batimentos, pressão, oxigenação) | `anomaly_detection/vitals.py` — 3 técnicas complementares: **regras fisiológicas** (faixas clínicas), **z-score móvel 3.5σ** (mudança vs. linha de base do paciente), **IsolationForest** (padrão multivariado atípico) | `pytest tests/test_vitals.py -v` ou dashboard → "Executar ciclo" (dessaturação SpO₂ 96→85%, taquicardia 135 bpm, pico hipertensivo 185 mmHg plantados nos dados sintéticos) |
| ✅ | Evolução de prescrições (alterações inesperadas) | `anomaly_detection/prescriptions.py` — dose acima da máxima de referência, salto abrupto de dose (>2x), interações medicamentosas conhecidas, reintrodução de suspenso | `pytest tests/test_prescriptions.py -v`; demo detecta dipirona 1000→3000 mg e warfarina+ibuprofeno |
| ✅ | Padrões de movimentação do paciente | `anomaly_detection/movement.py` — imobilidade prolongada (risco de úlcera/TVP) e picos (queda/agitação) | `pytest tests/test_movement.py -v` |
| ✅ | Alertas automáticos para a equipe médica | `integration/fusion.py` (fusão multimodal ponderada + bônus de corroboração → score de risco → `Alert`) + `integration/alerts.py` (3 canais: console, arquivo JSONL, **webhook Teams/Slack**) + campo `hypotheses` de apoio à decisão | `python scripts/run_demo.py` → alerta CRÍTICO no console; `ALERT_CHANNEL=webhook` + URL → alerta chega no Teams/Slack |

## 4. Requisitos transversais

| ✔ | Requisito | Implementação |
|---|---|---|
| ✅ | Fusão multimodal (texto + áudio + vídeo) | Late fusion com pesos clínicos por modalidade e bônus de corroboração (`integration/fusion.py`); múltiplas modalidades concordando elevam o score |
| ✅ | Integração com Azure Cognitive Services | Azure Speech to Text + Azure Language (Text Analytics), com credenciais via `.env` (nunca commitadas — `.env.example` documenta) |
| ✅ | Alertas em tempo real / near real-time | API REST FastAPI (`/monitor`, `/monitor/vitals`, `/monitor/prescriptions`, `/intake`) + despacho imediato por webhook |
| ✅ | Datasets | Dados sintéticos reprodutíveis (seed fixa) com anomalias plantadas — `synthetic.py` + `scripts/generate_synthetic_data.py`. PhysioNet/AudioSet citados e suportados: os detectores aceitam qualquer `DataFrame`/WAV/MP4 no formato documentado (`data/README.md`) |

## 5. Entregáveis

| ✔ | Entregável | Localização |
|---|---|---|
| ✅ | Código-fonte completo em repositório Git | Este repositório — histórico de ~20 commits Conventional Commits cobrindo todo o ciclo de desenvolvimento |
| ✅ | Relatório técnico: fluxo multimodal, modelos por tipo de dado, resultados e exemplos de anomalias | [`docs/relatorio_tecnico.md`](relatorio_tecnico.md) (arquitetura §2, modelos §3, fusão §4, resultados com tabela de anomalias detectadas §6) |
| ⬜ | Vídeo demonstrativo (até 15 min, YouTube/Vimeo) | *(inserir URL após gravação — roteiro sugerido no relatório técnico §9)* |

### Cobertura exigida no vídeo demonstrativo

| ✔ | Item exigido | Como demonstrar (sugestão) |
|---|---|---|
| ⬜ | Exemplo prático da análise de áudio e vídeo | Tela `/intake`: gravar voz ao microfone (transcrição Azure + termos críticos realçados) e enviar vídeo real (pose + YOLO) |
| ⬜ | Detecção e resposta a anomalias | Dashboard: alternar cenário "estável" ↔ "crítico" e mostrar o score/achados mudando |
| ⬜ | Integração dos serviços Azure | `GET /health` com `azure_speech: true` e `azure_language: true`; transcrição retornando `source: "azure"` |
| ⬜ | Fluxo final do alerta à equipe médica | Alerta CRÍTICO no banner do dashboard + hipóteses de apoio à decisão + (opcional) chegada no webhook Teams/Slack |

## 6. Extras implementados (além do exigido)

- **Dashboard clínico web interativo** (`/`) — gauge de risco, sparklines de sinais vitais, feed de achados filtrável por severidade, transcrição com realce de termos.
- **Coorte de 20 pacientes** (`/patients`) — seleção com busca/filtro e análise automática "amarrada" ao ID (vitais + áudio + vídeo + prescrições), com **player de vídeo real** e **áudio TTS pt-BR** por paciente.
- **Tela de captura clínica** (`/intake`) — formulário multimodal com gravação ao vivo de microfone (WAV 16 kHz)/webcam e resultado imediato.
- **Hipóteses interpretativas** no alerta — apoio à decisão por regras ("compatível com: ..."), com disclaimer explícito de que não substituem avaliação médica.
- **Modelagem clínica calibrada** — significância estatística **e** clínica (z-score com variação absoluta mínima; IsolationForest gated por faixa normal): estável → 0.00, crítico → 1.00.
- **Degradação graciosa** — sem GPU/credenciais Azure, todo o pipeline roda offline (modo mock), garantindo reprodutibilidade da correção.
- **Deploy em nuvem** — `Dockerfile` + guia para Azure Container Apps (`docs/deploy_azure.md`).
- **Qualidade de engenharia** — 41 testes automatizados (pytest), lint (ruff), type hints (mypy), CI GitHub Actions em Python 3.10–3.12.

---

## Guia rápido de correção (5 minutos)

```bash
# 1. Setup
python -m venv .venv && .venv\Scripts\activate    # Windows
pip install -r requirements.txt

# 2. Suite de testes (41 casos)
pytest

# 3. Demonstração ponta-a-ponta no terminal
python scripts/run_demo.py
#    → alerta CRÍTICO com 4 modalidades, score 1.00

# 4. Telas web + API
uvicorn multimodal_monitor.api.main:app --app-dir src
#    → http://127.0.0.1:8000/          (dashboard de monitoramento)
#    → http://127.0.0.1:8000/patients  (coorte: escolher paciente e analisar)
#    → http://127.0.0.1:8000/intake    (captura clínica multimodal)
#    → http://127.0.0.1:8000/docs      (Swagger)
```

> Para testar a integração Azure real: copiar `.env.example` → `.env` e
> preencher as chaves (instruções no README). Sem elas, o sistema opera em
> modo offline sem perda de funcionalidade demonstrável.
