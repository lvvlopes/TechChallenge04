# Monitoramento Multimodal de Pacientes 🏥

> **FIAP Pós-Tech IADT — Tech Challenge Fase 4**
> Sistema de IA para monitoramento contínuo de pacientes hospitalares por meio
> de dados multimodais (**áudio**, **vídeo** e **texto**), com detecção de
> anomalias em tempo real, geração de alertas para a equipe médica e
> **dashboard clínico web interativo**.

[![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-21%20passing-brightgreen)
![Ruff](https://img.shields.io/badge/lint-ruff-46a4b8)

---

## 📌 Visão geral

O sistema evolui um hospital que já usa IA para exames/documentos rumo ao
**monitoramento contínuo e multimodal**, cruzando três fontes de dados:

| Requisito | Fonte | Técnicas |
| --------- | ----- | -------- |
| **RF01 — Vídeo** | Cirurgias / fisioterapia | MediaPipe Pose (alternativa prática ao OpenPose), YOLOv8 |
| **RF02 — Áudio** | Consultas médicas | Azure Speech to Text + Azure Text Analytics (com fallback léxico offline) |
| **RF03 — Anomalias** | Sinais vitais, prescrições, movimentação | Regras fisiológicas, z-score móvel (3.5σ), IsolationForest |

Os resultados de cada modalidade são combinados por uma camada de **fusão
tardia (late fusion)** que produz um **score de risco** único e dispara alertas
clínicos quando o risco ultrapassa um limiar configurável.

```
┌──────────┐   ┌──────────┐   ┌──────────────┐
│  Vídeo   │   │  Áudio   │   │ Sinais vitais│
│ (RF01)   │   │ (RF02)   │   │ /prescr(RF03)│
└────┬─────┘   └────┬─────┘   └──────┬───────┘
     │ achados      │ achados        │ achados
     └──────────────┴────────────────┘
                    ▼
          ┌───────────────────┐
          │  Fusão multimodal │  → score de risco [0,1] + bônus de corroboração
          │  (late fusion)    │
          └─────────┬─────────┘
                    ▼
         ┌────────────────────────┐
         │  Alerta clínico         │  → console / arquivo / webhook (Teams/Slack)
         │  + Dashboard interativo │  → estação clínica web (FastAPI /)
         └────────────────────────┘
```

## ✨ Destaques de engenharia

- **Dashboard clínico web** com design profissional — gauge SVG de risco,
  sparklines Chart.js dos sinais vitais, feed de achados filtrável por
  severidade, painel de transcrição com termos críticos realçados e status
  em tempo real das integrações Azure/visão.
- **Degradação graciosa**: sem GPU, sem MediaPipe/YOLO e sem credenciais Azure,
  o sistema roda 100% offline em modo *mock* — ideal para demonstração e CI.
- **Modelagem clínica calibrada**: `risk_score` da modalidade capado pela
  severidade máxima presente (achados só MEDIUM não escalam a CRITICAL por
  acúmulo), z-score em 3.5σ (padrão da literatura), IsolationForest com
  `contamination='auto'` + cutoff por quantil.
- **Dados sintéticos reprodutíveis** com anomalias plantadas (seed fixa) e
  ruído calibrado para variabilidade típica de UTI.
- **API REST** (FastAPI) para monitoramento *near real-time*.
- **Suíte de testes** (pytest, 21 casos) cobrindo detectores, fusão e
  fluxo ponta-a-ponta.
- **Qualidade**: `ruff` (lint/format), `mypy` (tipos), CI no GitHub Actions
  em Python 3.10 · 3.11 · 3.12.

## 🚀 Início rápido

### 1. Pré-requisitos
- Python **3.10+**

### 2. Instalação

```bash
# ambiente virtual
python -m venv .venv
# Windows:  .venv\Scripts\activate     |  Linux/Mac:  source .venv/bin/activate

# dependências de runtime
pip install -r requirements.txt

# (opcional) instalar o pacote em modo editável + ferramentas de dev
pip install -e ".[dev]"
```

> **Dependências opcionais** (pesadas). Instale só se for processar mídia real:
> ```bash
> pip install ".[vision]"   # OpenCV + MediaPipe + YOLOv8 (RF01)
> pip install ".[audio]"    # Azure Speech + Text Analytics (RF02)
> ```

### 3. Configuração (opcional — só para usar Azure de verdade)

```bash
cp .env.example .env       # e preencha as chaves Azure
```
Sem `.env`, os módulos de áudio operam em modo offline (mock) usando um
analisador léxico pt-BR embutido e transcrições de referência em `.txt`.

### 4. Rodar a demonstração via CLI

```bash
python scripts/run_demo.py
# ou, com o pacote instalado:
mmonitor demo
```

A demo gera dados sintéticos com anomalias nas 4 fontes, executa o pipeline
multimodal e imprime o alerta clínico resultante (score de risco = **1.00**,
severidade **CRÍTICA**, 4 modalidades corroborando).

### 5. Subir a API + dashboard interativo

```bash
uvicorn multimodal_monitor.api.main:app --reload --app-dir src
```

- **Dashboard interativo**: <http://127.0.0.1:8000/> — estação clínica web
  com gauge de risco multimodal, sparklines dos sinais vitais, feed de
  achados filtrável, painel de transcrição da consulta com **termos críticos
  realçados inline** por severidade, banner de alerta que muda de cor pela
  severidade e status em tempo real das integrações Azure/visão.
  Botão **Executar ciclo** dispara `/demo/run` e alterna entre os cenários
  **crítico** e **estável** — visualmente distintos:
  - **Crítico**: score `1.00`, alerta CRITICAL, 4 modalidades contribuindo.
  - **Estável**: score `~0.75`, alerta HIGH, só vitais residuais.
- **Docs interativas da API** (Swagger): <http://127.0.0.1:8000/docs>

Exemplo de chamada direta à API:

```bash
curl -X POST http://127.0.0.1:8000/monitor \
  -H "Content-Type: application/json" \
  -d '{
        "patient_id": "PAC-001",
        "transcript": "Estou com muita falta de ar e dor no peito.",
        "readings": [
          {"timestamp": "2026-07-03T10:00:00Z", "spo2": 84, "heart_rate": 128}
        ]
      }'
```

## 📡 Endpoints da API

| Método | Rota | Descrição |
| ------ | ---- | --------- |
| `GET`  | `/`                        | Dashboard clínico interativo (HTML) |
| `GET`  | `/dashboard`               | Alias do dashboard |
| `GET`  | `/health`                  | Status + capacidades ativas (Azure/visão) |
| `GET`  | `/docs`                    | Swagger UI |
| `POST` | `/monitor`                 | Ciclo multimodal (vitais + prescrições + texto) |
| `POST` | `/monitor/vitals`          | Analisa apenas uma janela de sinais vitais |
| `POST` | `/monitor/prescriptions`   | Analisa apenas a evolução de prescrições |
| `POST` | `/demo/run`                | Executa um ciclo com dados sintéticos (`critical` \| `stable`) — payload rico com relatório + série de vitais + insights do áudio (usado pelo dashboard) |

## 🗂️ Estrutura do projeto

```
src/multimodal_monitor/
├── config.py             # settings via pydantic-settings (.env)
├── schemas.py            # contrato de domínio (Finding, Alert, Severity…)
├── anomaly_detection/    # RF03: vitals, prescriptions, movement
├── audio_analysis/       # RF02: speech-to-text, text analytics, pipeline
├── video_analysis/       # RF01: pose (MediaPipe), objetos (YOLOv8), pipeline
├── integration/          # fusão multimodal, alertas, orquestrador
├── api/
│   ├── main.py           # FastAPI: /monitor, /demo/run, /health, dashboard
│   └── static/
│       └── dashboard.html  # estação clínica web (single-file)
├── synthetic.py          # geradores de dados sintéticos calibrados para UTI
├── demo.py               # cenário de demonstração ponta-a-ponta
└── cli.py                # CLI `mmonitor`
scripts/                  # geração de dados / execução da demo
tests/                    # pytest (21 casos)
docs/relatorio_tecnico.md # relatório técnico (entregável)
notebooks/                # exploração multimodal
data/samples/             # datasets sintéticos reprodutíveis (versionados)
```

## 🧪 Desenvolvimento

```bash
pytest                                        # testes (21 casos)
ruff check .                                  # lint
ruff format .                                 # formatação
mypy src                                      # checagem de tipos
python scripts/generate_synthetic_data.py     # (re)gera data/samples/
```

## 🧠 Como as anomalias são detectadas (RF03)

| Método | O que captura | Exemplo |
| ------ | ------------- | ------- |
| **Regras fisiológicas** | Valores fora de faixas clínicas seguras | SpO2 < 88% ⇒ CRÍTICO |
| **Z-score móvel (3.5σ)** | Mudança abrupta vs. linha de base do paciente | FC salta 3.5σ em minutos |
| **IsolationForest** | Combinação multivariada atípica | FC normal + PA normal, mas *padrão* raro (top ~1% dos scores) |
| **Regras de prescrição** | Dose > máx., salto de dose, interações | Warfarina + Ibuprofeno |
| **Movimentação** | Imobilidade prolongada / picos | Queda (pico de movimento) |

### Como a modalidade compõe seu `risk_score`

O score de cada modalidade combina severidade × confiança dos achados, com
peso decrescente (`0.5^i`) para evitar saturação por volume — **e é capado
pela severidade máxima presente**. Consequência clínica: um paciente que
gera vários achados MEDIUM não tem seu risco escalado a CRITICAL só por
acúmulo; para chegar em CRITICAL é preciso ao menos um achado CRITICAL.

### Como a fusão multimodal combina as modalidades

Média ponderada dos `risk_scores` (vitais 1.0, prescrições 0.9, vídeo 0.7,
áudio/movimento 0.6) + **bônus de corroboração** por cada modalidade
adicional em risco relevante (≥0.25). Score final saturado em `[0,1]`,
mapeado para severidade e comparado ao limiar de alerta (`0.35` por
padrão).

## 🔒 Segurança & privacidade

- Credenciais Azure vivem apenas no `.env` (no `.gitignore`), **nunca** no código.
- Pastas `data/raw/` e `data/private/` são ignoradas — **não versione dados
  reais de pacientes**. Use apenas dados públicos/sintéticos.
- Os samples em `data/samples/` são inteiramente sintéticos e reprodutíveis
  via `scripts/generate_synthetic_data.py`.

## 📦 Entregáveis da Fase 4

- ✅ **Código-fonte** completo (este repositório) com histórico Git coerente.
- ✅ **Relatório técnico** — [`docs/relatorio_tecnico.md`](docs/relatorio_tecnico.md).
- ✅ **Dashboard clínico interativo** — servido pela FastAPI em `/`.
- ⬜ **Vídeo demonstrativo** (até 15 min) — a gravar (roteiro na seção 9 do relatório).

## 🧾 Histórico do projeto (visão de commits)

O histórico segue Conventional Commits e cobre todo o ciclo:

| Commit | Escopo |
| ------ | ------ |
| `chore: project scaffolding, tooling and CI` | Fundação: `pyproject`, requirements, Makefile, ruff/mypy, CI |
| `feat(core): domain contract, config and logging` | `schemas`, `config`, logging |
| `feat(anomaly): detecção de anomalias RF03` | vitals + prescriptions + movement |
| `feat(audio): análise de áudio RF02` | Azure Speech + Text Analytics + fallback offline |
| `feat(video): análise de vídeo RF01` | MediaPipe Pose + YOLOv8 + modo offline |
| `feat(integration): fusão multimodal, alertas e orquestração` | Late fusion + AlertManager + orquestrador |
| `feat(api,cli): API REST FastAPI e CLI mmonitor` | Endpoints + CLI |
| `feat(demo): dados sintéticos e cenário de demonstração` | Geradores reprodutíveis + `demo` |
| `test: suíte pytest (detectores, fusão, ponta-a-ponta)` | 21 casos |
| `docs: README, relatório técnico e notebook de exploração` | Documentação |
| `chore(ruff): exclui notebooks do lint` | Config de qualidade |
| `refactor(anomaly): calibra defaults clínicos e cap por severidade` | Modelagem afinada (3.5σ, cap por severidade máx., etc.) |
| `feat(dashboard): estação clínica web interativa` | Dashboard HTML/JS/CSS + endpoint `/demo/run` |
| `docs(readme): documenta dashboard interativo` | README |
| `chore(data): snapshot dos samples sintéticos calibrados` | Datasets de referência versionados |

## 📄 Licença

MIT — ver [`LICENSE`](LICENSE).
