# Relatório Técnico — Monitoramento Multimodal de Pacientes

**Curso:** FIAP Pós-Tech IADT · **Fase:** 4 · **Tech Challenge**
**Tema:** Monitoramento contínuo e multimodal de pacientes hospitalares com
detecção de anomalias em tempo real.

---

## 1. Resumo executivo

Este projeto implementa um sistema de IA que monitora pacientes hospitalares
cruzando três modalidades de dados — **vídeo** (postura/movimento em cirurgias e
fisioterapia), **áudio** (fala do paciente em consultas) e **texto/séries
temporais** (sinais vitais e prescrições). Cada modalidade é processada por um
pipeline dedicado que emite *achados* (`Finding`) padronizados; uma camada de
**fusão multimodal** combina esses achados em um **score de risco** único e
dispara **alertas clínicos** quando o risco ultrapassa um limiar configurável.

O sistema foi projetado com **degradação graciosa**: quando bibliotecas de visão
computacional (MediaPipe/YOLOv8) ou credenciais Azure não estão disponíveis, ele
opera em modo *offline/mock*, permitindo demonstração reprodutível e execução em
CI sem GPU nem custo de nuvem.

## 2. Arquitetura da solução

### 2.1 Visão geral

```
        RF01 (vídeo)            RF02 (áudio)         RF03 (vitais/prescrições)
   ┌──────────────────┐   ┌──────────────────┐   ┌────────────────────────┐
   │ MediaPipe Pose   │   │ Azure Speech STT │   │ Regras fisiológicas    │
   │ YOLOv8 (objetos) │   │ Azure Text Anal. │   │ Z-score móvel          │
   │                  │   │ (léxico offline) │   │ IsolationForest        │
   └────────┬─────────┘   └────────┬─────────┘   │ Regras de prescrição   │
            │                      │             │ Movimentação           │
            │  ModalityResult      │             └───────────┬────────────┘
            │  (lista de Finding)  │                         │
            └──────────────────────┴─────────────────────────┘
                                   ▼
                    ┌───────────────────────────┐
                    │   MultimodalFusion         │
                    │   • média ponderada        │
                    │   • bônus de corroboração  │
                    │   → risk_score ∈ [0,1]     │
                    └─────────────┬──────────────┘
                                  ▼
                    ┌───────────────────────────┐
                    │   AlertManager             │
                    │   console | file | webhook │
                    └───────────────────────────┘
```

### 2.2 Contrato de domínio

Todas as modalidades convergem para um vocabulário comum (`src/multimodal_monitor/schemas.py`):

- **`Finding`** — um achado individual (modalidade, severidade, descrição,
  `score ∈ [0,1]`, timestamp, metadados).
- **`ModalityResult`** — agrega os achados de uma modalidade e expõe um
  `risk_score` derivado (o achado mais grave domina; os demais somam com peso
  decrescente, evitando saturação por volume de achados).
- **`Severity`** — `INFO < LOW < MEDIUM < HIGH < CRITICAL`, com peso numérico
  usado na fusão.
- **`Alert`** — alerta clínico consolidado com os achados que mais contribuíram.

Esse contrato desacopla os pipelines: cada modalidade pode evoluir
independentemente desde que produza `Finding`s.

## 3. Modelos e técnicas por modalidade

### 3.1 RF01 — Análise de vídeo

**Objetivo:** detectar movimentos/eventos fora do padrão em vídeos clínicos.

- **Análise postural** (`pose_analyzer.py`): usamos **MediaPipe Pose** como
  alternativa prática e leve ao OpenPose (mesma finalidade: estimação de
  landmarks corporais). Por quadro amostrado derivamos:
  - `movement_index` — magnitude do deslocamento dos landmarks entre quadros
    consecutivos, normalizada em [0,1];
  - `trunk_angle` — inclinação do tronco (ombros→quadris) em relação à vertical.
- **Detecção de objetos/áreas críticas** (`object_detector.py`): **YOLOv8**
  (`ultralytics`) para identificar objetos relevantes no contexto clínico.
- **Achados gerados**:
  - **Imobilidade prolongada** (risco de úlcera de pressão/TVP) e **picos de
    movimento** (possível queda/agitação), via o detector de movimentação (§3.3);
  - **Desvio postural** quando a inclinação de tronco excede 35° (HIGH ≥ 55°).

> Em ambiente sem OpenCV/MediaPipe, o pipeline aceita **sinais de pose
> pré-computados** (`PoseFrame`), preservando toda a lógica de detecção.

### 3.2 RF02 — Análise de áudio

**Objetivo:** transcrever a fala do paciente e detectar termos críticos e
alterações vocais/sentimento indicativos de risco.

- **Transcrição** (`speech_to_text.py`): **Azure Speech to Text** (reconhecimento
  contínuo, `pt-BR`). Sem credenciais, o modo *mock* lê uma transcrição de
  referência de um arquivo `.txt` irmão do áudio.
- **Análise de texto** (`text_analytics.py`): **Azure Text Analytics** para
  sentimento; sem credenciais, um **analisador léxico offline** em pt-BR
  (dicionários de polaridade + termos críticos com normalização de acentos).
- **Achados gerados**:
  - **Termos críticos** ("dor no peito" ⇒ CRÍTICO, "falta de ar" ⇒ HIGH,
    "cansaço/fadiga" ⇒ MEDIUM, …);
  - **Sentimento negativo acentuado** (possível sofrimento/piora clínica).

### 3.3 RF03 — Detecção de anomalias

**a) Sinais vitais** (`vitals.py`) — três abordagens complementares:

1. **Regras fisiológicas** — faixas de segurança por sinal (FC, PA sistólica/
   diastólica, SpO2, FR, temperatura). Interpretáveis e sempre ativas.
2. **Z-score móvel** — desvio relativo a uma janela recente (linha de base do
   próprio paciente), capturando **mudanças abruptas**.
3. **IsolationForest** (multivariado) — captura **combinações atípicas** de
   sinais que, isoladamente, pareceriam normais.

**Significância estatística E clínica.** Para evitar falsos positivos por ruído
(um evento estatisticamente raro mas fisiologicamente irrelevante), os métodos
estatísticos exigem *duas* condições simultâneas: o z-score só sinaliza se, além
de `|z| ≥ 3.5`, a **variação absoluta** superar um mínimo clínico por sinal
(ex.: 15 bpm de FC, 3% de SpO2); o IsolationForest só reporta o padrão
multivariado se ao menos um sinal estiver **fora da faixa normal**. Resultado:
paciente estável → score 0.00 (sem alerta); crítico → 1.00.

**b) Prescrições** (`prescriptions.py`) — regras clínicas sobre a evolução:
dose acima da máxima diária de referência, **salto de dose** acima de fator
seguro, **interações medicamentosas** conhecidas e reintrodução de medicamento
suspenso.

**c) Movimentação** (`movement.py`) — sobre um índice de atividade temporal:
**imobilidade prolongada** e **picos** (queda/crise). Reutilizado pelo RF01.

> As bases de doses máximas e interações são **ilustrativas** (demonstram o
> mecanismo) e não substituem uma base farmacológica oficial.

## 4. Fusão multimodal

A fusão (`integration/fusion.py`) usa **late fusion**:

1. Cada modalidade contribui com seu `risk_score`, ponderado por um **peso
   clínico** (vitais 1.0, prescrições 0.9, vídeo 0.7, áudio/movimento 0.6).
2. Calcula-se a **média ponderada** dos scores ativos.
3. Aplica-se um **bônus de corroboração**: quando várias modalidades concordam
   em risco relevante, a confiança aumenta (o todo é maior que a soma das
   partes) — refletindo a intuição clínica de sinais convergentes.
4. O score final (saturado em [0,1]) é mapeado para uma `Severity` e, acima do
   limiar (`0.35`), gera um `Alert` com os achados mais graves de cada fonte.

## 5. Alertas em tempo real

O `AlertManager` (`integration/alerts.py`) despacha alertas por três canais
configuráveis via `ALERT_CHANNEL`:

- **console** — painel formatado (rich) para operação/demonstração;
- **file** — `outputs/alerts.jsonl` (auditoria/persistência);
- **webhook** — POST JSON para Microsoft Teams / Slack (integração real).

Cada `Alert` carrega ainda **hipóteses interpretativas de apoio à decisão**
(`integration/hypotheses.py`): regras que mapeiam padrões dos achados para
possíveis causas ("compatível com: …"), incluindo combinações multimodais
(ex.: *dispneia relatada + dessaturação objetiva → priorizar avaliação
respiratória*). **Não é diagnóstico** — todas as hipóteses vêm com o aviso de
que não substituem a avaliação médica, mantendo o sistema na categoria de
apoio à decisão clínica (CDSS).

A **API REST** (`api/main.py`, FastAPI) expõe endpoints para monitoramento
*near real-time* (`/monitor`, `/monitor/vitals`, `/monitor/prescriptions`,
`/intake`), a coorte de pacientes (`/api/patients`, `/api/patients/{id}/analyze`)
e um `/health` que reporta as capacidades disponíveis (Azure/visão). Três telas
web (dashboard `/`, coorte `/patients`, captura `/intake`) tornam a operação
visual.

### 5.1 Coorte de pacientes e "amarração" por ID

Uma coorte de 20 pacientes sintéticos é persistida em `data/patients.json`
(binários em `data/patients_media/`), cada um
com dados coerentes com seu cenário (estável/crítico) e "amarrados" ao ID:
sinais vitais, prescrições, **áudio real** (sintetizado por TTS em pt-BR e
gravado em WAV 16 kHz mono, transcrito pelo Azure Speech quando ativo) e sinais
de pose. Ao selecionar um paciente na tela `/patients`, o sistema carrega
automaticamente todos esses dados e executa o pipeline multimodal. Pacientes
com um `video_teste.mp4` na pasta têm a **pose extraída do vídeo real** e o
vídeo é exibido na tela.

## 6. Resultados e exemplos de anomalias detectadas

O cenário de demonstração (`mmonitor demo`) gera dados sintéticos com anomalias
plantadas nas quatro fontes. Resultado observado:

| Modalidade | Risco | Achados | Exemplo de anomalia detectada |
| ---------- | ----- | ------- | ----------------------------- |
| Sinais vitais | 1.00 | ~50 | Dessaturação de SpO2 (96→85%), taquicardia (135 bpm), pico hipertensivo (185 mmHg) |
| Prescrições | 1.00 | 3 | Salto de dose de dipirona (1000→3000 mg); interação warfarina + ibuprofeno |
| Áudio | 1.00 | 4–5 | Termos "dor no peito" (CRÍTICO) e "falta de ar" (HIGH); sentimento negativo |
| Vídeo | 1.00 | 8–46 | Desvio postural, imobilidade e picos (8 na pose sintética; ~46 no vídeo real) |

**Score de risco multimodal = 1.00 → alerta CRÍTICO** despachado com mensagem:

> *"Sinais de risco corroborados por 4 modalidade(s) (vitals, prescription,
> audio, video). Principais achados: Frequência cardíaca acima do esperado;
> Pressão sistólica acima do esperado; Termo crítico 'dor no peito'; Pico
> abrupto de movimentação…"*

A suíte de testes (`pytest`, 41 casos) valida cada detector, a fusão (incluindo
o efeito de corroboração e limites do score) e o **fluxo ponta-a-ponta**.

## 7. Reprodutibilidade

```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
python scripts/run_demo.py        # demonstração completa
pytest                            # suíte de testes
```

Todos os geradores sintéticos aceitam `seed`, garantindo resultados
determinísticos. O CI (GitHub Actions) executa lint + testes em Python 3.10–3.12.

## 8. Limitações e trabalhos futuros

- **Bases clínicas ilustrativas** (doses/interações): integrar a uma base
  farmacológica oficial (ex.: bulário/ANVISA) em produção.
- **Fusão**: evoluir de *late fusion* por regras para um modelo aprendido
  (ex.: gradient boosting sobre features multimodais) quando houver dados
  rotulados de desfechos clínicos.
- **Áudio**: além de STT+sentimento, extrair *biomarcadores vocais* (jitter,
  shimmer, F0) para fadiga/dispneia diretamente do sinal.
- **Vídeo**: rastreamento temporal por paciente e ações específicas
  (protocolos de fisioterapia) com modelos de ação.
- **Tempo real**: streaming (WebSocket/Kafka) e janelas deslizantes com estado
  por paciente.

## 9. Roteiro sugerido para o vídeo demonstrativo (até 15 min)

1. **Contexto** (1 min) — problema clínico e proposta multimodal.
2. **Arquitetura** (2 min) — diagrama do fluxo e contrato de domínio.
3. **RF02 — Áudio** (3 min) — transcrição + termos críticos + sentimento (mock e/ou Azure).
4. **RF01 — Vídeo** (3 min) — pose/movimento e desvio postural.
5. **RF03 — Anomalias** (3 min) — sinais vitais (regras + z-score + IsolationForest) e prescrições.
6. **Fusão + Alerta** (2 min) — `mmonitor demo` gerando o alerta CRÍTICO; integração Azure e webhook.
7. **Encerramento** (1 min) — resultados, limitações e próximos passos.

---

## Anexo A — Mapa Requisito → Código

| Requisito | Implementação |
| --------- | ------------- |
| RF01 Vídeo | `video_analysis/{pose_analyzer,object_detector,video_pipeline}.py` |
| RF02 Áudio | `audio_analysis/{speech_to_text,text_analytics,audio_pipeline}.py` |
| RF03 Anomalias | `anomaly_detection/{vitals,prescriptions,movement}.py` |
| Fusão multimodal | `integration/fusion.py` |
| Alertas em tempo real | `integration/alerts.py`, `api/main.py` |
| Orquestração | `integration/orchestrator.py` |
| Azure Cognitive Services | `audio_analysis/*` (Speech + Text Analytics) |
