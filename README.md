# Monitoramento Multimodal de Pacientes 🏥

> **FIAP Pós-Tech IADT — Tech Challenge Fase 4**
> Sistema de IA para monitoramento contínuo de pacientes hospitalares por meio
> de dados multimodais (**áudio**, **vídeo** e **texto**), com detecção de
> anomalias em tempo real e geração de alertas para a equipe médica.

[![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📌 Visão geral

O sistema evolui um hospital que já usa IA para exames/documentos rumo ao
**monitoramento contínuo e multimodal**, cruzando três fontes de dados:

| Requisito | Fonte | Técnicas |
| --------- | ----- | -------- |
| **RF01 — Vídeo** | Cirurgias / fisioterapia | MediaPipe Pose (alternativa prática ao OpenPose), YOLOv8 |
| **RF02 — Áudio** | Consultas médicas | Azure Speech to Text + Azure Text Analytics |
| **RF03 — Anomalias** | Sinais vitais, prescrições, movimentação | Regras fisiológicas, z-score móvel, IsolationForest |

Os resultados de cada modalidade são combinados por uma camada de **fusão
tardia (late fusion)** que produz um **score de risco** único e dispara alertas
clínicos quando o risco ultrapassa um limiar.

```
┌──────────┐   ┌──────────┐   ┌──────────────┐
│  Vídeo   │   │  Áudio   │   │ Sinais vitais│
│ (RF01)   │   │ (RF02)   │   │ /prescr(RF03)│
└────┬─────┘   └────┬─────┘   └──────┬───────┘
     │ achados      │ achados        │ achados
     └──────────────┴────────────────┘
                    ▼
          ┌───────────────────┐
          │  Fusão multimodal │  → score de risco [0,1]
          │  (late fusion)    │
          └─────────┬─────────┘
                    ▼
          ┌───────────────────┐
          │  Alerta clínico   │  → console / arquivo / webhook (Teams/Slack)
          └───────────────────┘
```

## ✨ Destaques de engenharia

- **Degradação graciosa**: sem GPU, sem MediaPipe/YOLO e sem credenciais Azure,
  o sistema roda 100% offline em modo *mock* — ideal para demonstração e CI.
- **Dados sintéticos reprodutíveis** com anomalias plantadas (seed fixa).
- **API REST** (FastAPI) para monitoramento *near real-time*.
- **Suíte de testes** (pytest) cobrindo detectores, fusão e o fluxo ponta-a-ponta.
- **Qualidade**: `ruff` (lint/format), `mypy` (tipos), CI no GitHub Actions.

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
Sem `.env`, os módulos de áudio operam em modo offline (mock).

### 4. Rodar a demonstração

```bash
python scripts/run_demo.py
# ou, com o pacote instalado:
mmonitor demo
```

A demo gera dados sintéticos com anomalias nas 4 fontes, executa o pipeline
multimodal e imprime o alerta clínico resultante (score de risco = 1.00,
severidade CRÍTICA).

### 5. Subir a API

```bash
uvicorn multimodal_monitor.api.main:app --reload --app-dir src
# docs interativas em  http://127.0.0.1:8000/docs
```

Exemplo de chamada:

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

## 🗂️ Estrutura do projeto

```
src/multimodal_monitor/
├── config.py             # settings via pydantic-settings (.env)
├── schemas.py            # contrato de domínio (Finding, Alert, Severity…)
├── anomaly_detection/    # RF03: vitals, prescriptions, movement
├── audio_analysis/       # RF02: speech-to-text, text analytics, pipeline
├── video_analysis/       # RF01: pose (MediaPipe), objetos (YOLOv8), pipeline
├── integration/          # fusão multimodal, alertas, orquestrador
├── api/                  # FastAPI
├── synthetic.py          # geradores de dados sintéticos
├── demo.py               # cenário de demonstração ponta-a-ponta
└── cli.py                # CLI `mmonitor`
scripts/                  # geração de dados / execução da demo
tests/                    # pytest
docs/relatorio_tecnico.md # relatório técnico (entregável)
data/                     # datasets (samples versionáveis; raw/private ignorados)
```

## 🧪 Desenvolvimento

```bash
pytest                 # testes
ruff check .           # lint
ruff format .          # formatação
mypy src               # checagem de tipos
python scripts/generate_synthetic_data.py   # (re)gera data/samples/
```

## 🧠 Como as anomalias são detectadas (RF03)

| Método | O que captura | Exemplo |
| ------ | ------------- | ------- |
| **Regras fisiológicas** | Valores fora de faixas clínicas seguras | SpO2 < 88% ⇒ CRÍTICO |
| **Z-score móvel** | Mudança abrupta vs. linha de base do paciente | FC salta 3σ em minutos |
| **IsolationForest** | Combinação multivariada atípica | FC normal + PA normal, mas *padrão* raro |
| **Regras de prescrição** | Dose > máx., salto de dose, interações | Warfarina + Ibuprofeno |
| **Movimentação** | Imobilidade prolongada / picos | Queda (pico de movimento) |

## 🔒 Segurança & privacidade

- Credenciais Azure vivem apenas no `.env` (no `.gitignore`), **nunca** no código.
- Pastas `data/raw/` e `data/private/` são ignoradas — **não versione dados
  reais de pacientes**. Use apenas dados públicos/sintéticos.

## 📦 Entregáveis da Fase 4

- ✅ **Código-fonte** completo (este repositório).
- ✅ **Relatório técnico** — [`docs/relatorio_tecnico.md`](docs/relatorio_tecnico.md).
- ⬜ **Vídeo demonstrativo** (até 15 min) — a gravar (roteiro no relatório).

## 📄 Licença

MIT — ver [`LICENSE`](LICENSE).
