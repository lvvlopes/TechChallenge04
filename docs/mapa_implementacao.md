# Mapa de Implementação — Requisito → Código-Fonte

**Tech Challenge Fase 4 — Monitoramento Multimodal de Pacientes**

Este documento rastreia cada item das *Entregas Técnicas* do enunciado até o
ponto exato do código (arquivo e função) que o implementa, com o comando de
verificação correspondente. Referências no formato `arquivo.py:linha`.

---

## 1. Análise de Vídeo

### 1.1 — Processar vídeos clínicos (fisioterapia / cirurgias gravadas)

| Onde | Detalhe |
|---|---|
| [`video_analysis/video_pipeline.py:45`](../src/multimodal_monitor/video_analysis/video_pipeline.py) | `VideoAnalysisPipeline.process(video_path)` — ponto de entrada da análise de um vídeo |
| [`video_analysis/pose_analyzer.py:92`](../src/multimodal_monitor/video_analysis/pose_analyzer.py) | `PoseAnalyzer.analyze_video()` — abre o vídeo com OpenCV (`cv2.VideoCapture`), itera os quadros amostrados |

**Verificação:**
```bash
python -c "from multimodal_monitor.video_analysis import VideoAnalysisPipeline; print(VideoAnalysisPipeline().process('data/samples/video_teste.mp4').summary)"
```

### 1.2 — Detectar movimentos/eventos fora do padrão

**Análise postural** (enunciado: "modelos como OpenPose"):

| Onde | Detalhe |
|---|---|
| [`video_analysis/pose_analyzer.py:34`](../src/multimodal_monitor/video_analysis/pose_analyzer.py) | `_load_pose_module()` — carrega **MediaPipe Pose** (33 landmarks corporais) |
| [`video_analysis/pose_analyzer.py:102`](../src/multimodal_monitor/video_analysis/pose_analyzer.py) | Extração de landmarks e cálculo de `movement_index` (deslocamento entre quadros) |
| [`video_analysis/pose_analyzer.py:135`](../src/multimodal_monitor/video_analysis/pose_analyzer.py) | `_trunk_angle()` — ângulo do tronco (ombros→quadris) para desvio postural |
| [`video_analysis/video_pipeline.py:93`](../src/multimodal_monitor/video_analysis/video_pipeline.py) | `_posture_findings()` — regra de desvio postural (≥ 35° = MEDIUM, ≥ 55° = HIGH) |

> **Nota:** MediaPipe Pose substitui OpenPose (mesma finalidade: estimação de
> pose por landmarks). Justificativa técnica no relatório, §3.1.

**YOLOv8** (detecção de objetos e áreas críticas):

| Onde | Detalhe |
|---|---|
| [`video_analysis/object_detector.py:29`](../src/multimodal_monitor/video_analysis/object_detector.py) | Classe `ObjectDetector` — wrapper do YOLOv8 |
| [`video_analysis/object_detector.py:50`](../src/multimodal_monitor/video_analysis/object_detector.py) | `from ultralytics import YOLO` — carrega o modelo `yolov8n.pt` |
| [`video_analysis/object_detector.py:55`](../src/multimodal_monitor/video_analysis/object_detector.py) | `detect_video()` — roda YOLOv8 nos quadros e retorna `Detection` (label, confiança, bbox) |

**Verificação:**
```bash
python -c "from multimodal_monitor.video_analysis import ObjectDetector; [print(d.label, round(d.confidence,2)) for d in ObjectDetector().detect_video('data/samples/video_teste.mp4', sample_rate=30)]"
```

### 1.3 — Gerar relatórios automáticos indicando desvios/falhas

| Onde | Detalhe |
|---|---|
| [`video_analysis/video_pipeline.py:58`](../src/multimodal_monitor/video_analysis/video_pipeline.py) | `_build_result()` — monta o `ModalityResult` com sumário + achados |
| [`schemas.py`](../src/multimodal_monitor/schemas.py) | `Finding` (severidade, descrição, score, timestamp, metadados) e `ModalityResult` (sumário + lista de achados) — o "relatório" estruturado |

**Verificação:** o retorno de `process()` já é o relatório; via API em `POST /intake` → campo `modalities.video`.

---

## 2. Análise de Áudio

### 2.1 — Processar áudios de consultas médicas

| Onde | Detalhe |
|---|---|
| [`audio_analysis/audio_pipeline.py`](../src/multimodal_monitor/audio_analysis/audio_pipeline.py) | `AudioAnalysisPipeline.process(audio)` — orquestra transcrição → análise → achados |

### 2.2 — Detectar alterações vocais indicativas de condições médicas

| Onde | Detalhe |
|---|---|
| [`audio_analysis/text_analytics.py:25`](../src/multimodal_monitor/audio_analysis/text_analytics.py) | `CRITICAL_TERMS` — dicionário clínico pt-BR: "cansaço/fadiga" (MEDIUM), "falta de ar", "dificuldade para respirar" (HIGH), "dor no peito" (CRITICAL), etc. |
| [`audio_analysis/text_analytics.py:109`](../src/multimodal_monitor/audio_analysis/text_analytics.py) | `_extract_critical_terms()` — varre a transcrição e sinaliza os termos |

**Verificação:**
```bash
python -c "from multimodal_monitor.audio_analysis import TextAnalytics; print(TextAnalytics().analyze('estou cansado e com falta de ar').critical_terms)"
```

### 2.3 — Azure Speech to Text para transcrever

| Onde | Detalhe |
|---|---|
| [`audio_analysis/speech_to_text.py:94`](../src/multimodal_monitor/audio_analysis/speech_to_text.py) | `_transcribe_azure()` — usa o SDK `azure.cognitiveservices.speech` (reconhecimento contínuo, pt-BR) |
| [`audio_analysis/speech_to_text.py:106`](../src/multimodal_monitor/audio_analysis/speech_to_text.py) | `speechsdk.SpeechRecognizer` — o reconhecedor real |
| [`audio_analysis/speech_to_text.py:41`](../src/multimodal_monitor/audio_analysis/speech_to_text.py) | `transcribe()` — decide entre Azure (se há credenciais) e fallback offline |
| [`config.py`](../src/multimodal_monitor/config.py) | `azure_speech_key` / `azure_speech_region` lidos do `.env` |

**Verificação:** `GET /health` → `capabilities.azure_speech: true`; transcrição real retorna `source: "azure"`.

### 2.4 — Termos críticos e sentimentos com Azure Text Analytics

| Onde | Detalhe |
|---|---|
| [`audio_analysis/text_analytics.py:131`](../src/multimodal_monitor/audio_analysis/text_analytics.py) | `_sentiment_azure()` — usa `TextAnalyticsClient.analyze_sentiment(..., language="pt")` |
| [`audio_analysis/text_analytics.py:112`](../src/multimodal_monitor/audio_analysis/text_analytics.py) | Termos críticos extraídos via `CRITICAL_TERMS` |
| [`audio_analysis/audio_pipeline.py`](../src/multimodal_monitor/audio_analysis/audio_pipeline.py) | Converte termos + sentimento negativo em `Finding`s |

**Verificação:** `GET /health` → `capabilities.azure_language: true`; dashboard realça termos por severidade no painel de transcrição.

---

## 3. Detecção de Anomalias

### 3.1a — Séries temporais de sinais vitais (batimentos, pressão, oxigenação)

| Onde | Detalhe |
|---|---|
| [`anomaly_detection/vitals.py:30`](../src/multimodal_monitor/anomaly_detection/vitals.py) | `PHYSIOLOGICAL_RANGES` — faixas clínicas por sinal (FC, PA sist./diast., SpO₂, FR, temperatura) |
| [`anomaly_detection/vitals.py:93`](../src/multimodal_monitor/anomaly_detection/vitals.py) | `_rule_based()` — **regras fisiológicas** (ex.: SpO₂ < 88% = CRÍTICO) |
| [`anomaly_detection/vitals.py:146`](../src/multimodal_monitor/anomaly_detection/vitals.py) | `_zscore()` — **z-score móvel 3.5σ** (mudança abrupta vs. linha de base) |
| [`anomaly_detection/vitals.py:182`](../src/multimodal_monitor/anomaly_detection/vitals.py) | `_isolation_forest()` — **IsolationForest** (padrão multivariado atípico) |

**Verificação:** `pytest tests/test_vitals.py -v`

### 3.1b — Evolução de prescrições (alterações inesperadas)

| Onde | Detalhe |
|---|---|
| [`anomaly_detection/prescriptions.py:91`](../src/multimodal_monitor/anomaly_detection/prescriptions.py) | `_check_max_dose()` — dose acima da máxima diária de referência |
| [`anomaly_detection/prescriptions.py:110`](../src/multimodal_monitor/anomaly_detection/prescriptions.py) | `_check_dose_jump()` — salto abrupto de dose (> 2×) |
| [`anomaly_detection/prescriptions.py:149`](../src/multimodal_monitor/anomaly_detection/prescriptions.py) | `_check_interactions()` — interações medicamentosas (`KNOWN_INTERACTIONS`, linha 37) |

**Verificação:** `pytest tests/test_prescriptions.py -v`

### 3.1c — Padrões de movimentação do paciente

| Onde | Detalhe |
|---|---|
| [`anomaly_detection/movement.py:41`](../src/multimodal_monitor/anomaly_detection/movement.py) | `_detect_immobility()` — imobilidade prolongada (risco de lesão por pressão/TVP) |
| [`anomaly_detection/movement.py:78`](../src/multimodal_monitor/anomaly_detection/movement.py) | `_detect_spikes()` — picos abruptos (queda/agitação) |

**Verificação:** `pytest tests/test_movement.py -v`

### 3.2 — Gerar alertas automáticos para a equipe médica

| Onde | Detalhe |
|---|---|
| [`integration/fusion.py:54`](../src/multimodal_monitor/integration/fusion.py) | `fuse()` — funde as modalidades num score de risco (média ponderada + bônus de corroboração, linha 69) |
| [`integration/fusion.py:89`](../src/multimodal_monitor/integration/fusion.py) | `_build_alert()` — cria o `Alert` (título, mensagem, achados, hipóteses) |
| [`integration/hypotheses.py`](../src/multimodal_monitor/integration/hypotheses.py) | `generate_hypotheses()` — hipóteses interpretativas de apoio à decisão |
| [`integration/alerts.py:42`](../src/multimodal_monitor/integration/alerts.py) | `dispatch()` — despacha por 3 canais: `_to_console` (53), `_to_file` (64), `_to_webhook` (71 — Teams/Slack) |

**Verificação:** `python scripts/run_demo.py` → alerta CRÍTICO no console.

---

## Apoio à decisão — hipóteses interpretativas

| Onde | Detalhe |
|---|---|
| [`integration/hypotheses.py`](../src/multimodal_monitor/integration/hypotheses.py) | `generate_hypotheses()` — mapeia padrões dos achados para hipóteses de causa possível ("compatível com: …"), com combos multimodais (ex.: dispneia relatada + dessaturação objetiva). Sempre com disclaimer de que não substituem avaliação médica. |
| [`integration/fusion.py`](../src/multimodal_monitor/integration/fusion.py) | O `Alert` carrega o campo `hypotheses`, preenchido a partir de todos os achados. |

## Coorte de pacientes — seleção e análise por ID

| Onde | Detalhe |
|---|---|
| [`cohort.py`](../src/multimodal_monitor/cohort.py) | `PatientCohort` — lê o manifesto `data/patients/cohort.json` e monta o `MonitoringInput` de um paciente a partir dos arquivos "amarrados" ao ID (vitais, prescrições, áudio, pose). |
| [`scripts/generate_patient_cohort.py`](../scripts/generate_patient_cohort.py) | Gera 20 pacientes (10 críticos, 10 estáveis) reprodutíveis. |
| [`scripts/generate_patient_audio.ps1`](../scripts/generate_patient_audio.ps1) | Gera o `consulta.wav` real de cada paciente (voz SAPI pt-BR + WAV 16 kHz mono). |
| [`scripts/extract_patient_pose.py`](../scripts/extract_patient_pose.py) | Extrai a pose real de `video_teste.mp4` (quando presente na pasta) para `pose_frames.csv`. |
| `api/main.py` | `GET /api/patients` (lista), `POST /api/patients/{id}/analyze` (análise por ID), `GET /api/patients/{id}/video` (serve o vídeo real). |

## Camada de integração e interfaces

| Componente | Onde | Papel |
|---|---|---|
| Orquestrador | [`integration/orchestrator.py`](../src/multimodal_monitor/integration/orchestrator.py) | `PatientMonitor.run()` — executa todas as modalidades e funde |
| Contrato de domínio | [`schemas.py`](../src/multimodal_monitor/schemas.py) | `Finding`, `ModalityResult`, `Alert` (com `hypotheses`), `Severity` |
| API REST | [`api/main.py`](../src/multimodal_monitor/api/main.py) | `/health`, `/monitor*`, `/demo/run`, `/intake`, `/api/patients*` |
| Dashboard web | [`api/static/dashboard.html`](../src/multimodal_monitor/api/static/dashboard.html) | Monitoramento (crítico × estável) |
| Coorte web | [`api/static/patients.html`](../src/multimodal_monitor/api/static/patients.html) | Seleção e análise dos 20 pacientes, com player de vídeo |
| Captura clínica | [`api/static/intake.html`](../src/multimodal_monitor/api/static/intake.html) | Formulário multimodal com gravação de mic/webcam |
| Dados sintéticos | [`synthetic.py`](../src/multimodal_monitor/synthetic.py) | Geradores reprodutíveis com anomalias plantadas |

---

## Resumo de rastreabilidade

```
1. VÍDEO
   1.1 processar vídeos ......... video_pipeline.py:45 + pose_analyzer.py:92
   1.2 análise postural ......... pose_analyzer.py (MediaPipe Pose)
   1.2 YOLOv8 objetos ........... object_detector.py:55
   1.3 relatórios ............... video_pipeline.py:58 + schemas.py (Finding)

2. ÁUDIO
   2.1 processar áudios ......... audio_pipeline.py
   2.2 alterações vocais ........ text_analytics.py:25 (CRITICAL_TERMS)
   2.3 Azure Speech to Text ..... speech_to_text.py:94
   2.4 Azure Text Analytics ..... text_analytics.py:131 (sentimento) + :112 (termos)

3. ANOMALIAS
   3.1 sinais vitais ............ vitals.py:93/146/182 (3 técnicas)
   3.1 prescrições .............. prescriptions.py:91/110/149
   3.1 movimentação ............. movement.py:41/78
   3.2 alertas .................. fusion.py:54 + alerts.py:42
```
