# Monitoramento Multimodal de Pacientes 🏥

> **FIAP Pós-Tech IADT — Tech Challenge Fase 4**
> Sistema de IA para monitoramento contínuo de pacientes hospitalares por meio
> de dados multimodais (**áudio**, **vídeo** e **texto**), com detecção de
> anomalias em tempo real, geração de alertas para a equipe médica e três telas
> web interativas (**monitoramento**, **coorte de pacientes** e **captura clínica**).

[![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-41%20passing-brightgreen)
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
tardia (late fusion)** que produz um **score de risco** único, dispara alertas
clínicos e gera **hipóteses interpretativas de apoio à decisão** (sem substituir
o médico) quando o risco ultrapassa um limiar configurável.

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
      ┌──────────────────────────────┐
      │  Alerta clínico + hipóteses   │  → console / arquivo / webhook (Teams/Slack)
      │  + 3 telas web interativas    │  → / (monitoramento) · /patients · /intake
      └──────────────────────────────┘
```

## ✨ Destaques de engenharia

- **Três telas web** com design profissional:
  - **`/`** — dashboard de monitoramento: gauge SVG de risco, sparklines dos
    sinais vitais, feed de achados filtrável, transcrição com termos realçados,
    bloco de hipóteses e status das integrações Azure/visão.
  - **`/patients`** — coorte de 20 pacientes: seleção com busca/filtro, análise
    automática "amarrada" ao ID (vitais + áudio + vídeo + prescrições), com
    player de vídeo real quando disponível.
  - **`/intake`** — captura clínica: formulário multimodal com **gravação ao vivo
    de microfone (WAV 16 kHz) e webcam**, upload de arquivos e resultado imediato.
- **Coorte de pacientes persistida** — 20 pacientes com cenários estável/crítico,
  cada um com áudio real (TTS pt-BR), sinais vitais, prescrições e pose.
- **Hipóteses de apoio à decisão** — regras interpretáveis ("compatível com: …"),
  sempre com aviso de que não substituem avaliação médica.
- **Degradação graciosa**: sem GPU, sem MediaPipe/YOLO e sem credenciais Azure,
  o sistema roda 100% offline em modo *mock* — ideal para demonstração e CI.
- **Modelagem clínica calibrada** — significância estatística **E** clínica:
  z-score exige variação absoluta mínima por sinal; IsolationForest só reporta
  padrão multivariado com ao menos um sinal fora da faixa normal; `risk_score`
  da modalidade capado pela severidade máxima presente.
- **Análise de vídeo isolada** — a extração de pose do vídeo real roda offline
  (evita conflitos protobuf/torch/mediapipe dentro do servidor web).
- **Deploy em nuvem** — Dockerfile + guia para Azure Container Apps.
- **Suíte de testes** (pytest, **41 casos**), `ruff`, `mypy` e CI (Python 3.10–3.12).

## 🚀 Início rápido

### 1. Pré-requisitos
- Python **3.10+**

### 2. Instalação

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate     |  Linux/Mac:  source .venv/bin/activate

pip install -r requirements.txt
pip install -e ".[dev]"          # (opcional) pacote editável + ferramentas dev
```

> **Dependências opcionais** (pesadas). Instale só se for processar mídia real:
> ```bash
> pip install ".[vision]"   # OpenCV + MediaPipe + YOLOv8 (RF01)
> pip install ".[audio]"    # Azure Speech + Text Analytics (RF02)
> ```

### 3. Configuração (opcional — só para usar Azure de verdade)

```bash
cp .env.example .env       # e preencha as chaves Azure (Speech + Language)
```
Sem `.env`, os módulos de áudio operam offline (analisador léxico pt-BR embutido
+ transcrições de referência `.txt`).

### 4. Rodar a demonstração via CLI

```bash
python scripts/run_demo.py         # ou: mmonitor demo
```
Gera dados sintéticos com anomalias nas 4 fontes e imprime o alerta clínico
resultante (score **1.00**, severidade **CRÍTICA**, 4 modalidades corroborando).

### 5. Subir a API + as telas interativas

```bash
uvicorn multimodal_monitor.api.main:app --reload --app-dir src
```

| Tela | URL | O que faz |
| ---- | --- | --------- |
| **Monitoramento** | <http://127.0.0.1:8000/> | Dashboard com cenário **crítico** (`1.00`) × **estável** (`0.00`) |
| **Pacientes** | <http://127.0.0.1:8000/patients> | Escolhe 1 dos 20 pacientes → análise automática por ID |
| **Nova admissão** | <http://127.0.0.1:8000/intake> | Formulário multimodal + gravação de mic/webcam |
| **Swagger** | <http://127.0.0.1:8000/docs> | Documentação interativa da API |

### 6. (Opcional) Gerar a coorte e as mídias

```bash
python scripts/generate_patient_cohort.py            # 20 pacientes em data/patients.json
powershell -File scripts/generate_patient_audio.ps1  # áudio TTS pt-BR por paciente (Windows)
# vídeo real de um paciente: copie o mp4 para a mídia e extraia a pose
cp data/samples/video_teste.mp4 data/patients_media/PAC-001.mp4
python scripts/extract_patient_pose.py
```

## 📡 Endpoints da API

| Método | Rota | Descrição |
| ------ | ---- | --------- |
| `GET`  | `/` · `/dashboard`         | Dashboard de monitoramento (HTML) |
| `GET`  | `/patients`               | Tela da coorte de pacientes (HTML) |
| `GET`  | `/intake`                 | Tela de captura clínica (HTML) |
| `GET`  | `/health`                 | Status + capacidades ativas (Azure/visão) |
| `GET`  | `/docs`                   | Swagger UI |
| `POST` | `/monitor`                | Ciclo multimodal (vitais + prescrições + texto) |
| `POST` | `/monitor/vitals`         | Analisa apenas sinais vitais |
| `POST` | `/monitor/prescriptions`  | Analisa apenas a evolução de prescrições |
| `POST` | `/demo/run`               | Ciclo com dados sintéticos (`critical` \| `stable`) — usado pelo dashboard |
| `POST` | `/intake`                 | Captura clínica multipart (sintomas + vitais + prescrições + upload áudio/vídeo) |
| `GET`  | `/api/patients`           | Lista os pacientes da coorte |
| `POST` | `/api/patients/{id}/analyze` | Analisa um paciente pelos dados "amarrados" ao ID |
| `GET`  | `/api/patients/{id}/video`   | Serve o vídeo real do paciente (quando presente) |

## 🗂️ Estrutura do projeto

```
src/multimodal_monitor/
├── config.py             # settings via pydantic-settings (.env)
├── schemas.py            # contrato de domínio (Finding, Alert, Severity…)
├── cohort.py             # coorte de pacientes: carrega dados por ID
├── synthetic.py          # geradores de dados sintéticos calibrados para UTI
├── anomaly_detection/    # RF03: vitals, prescriptions, movement
├── audio_analysis/       # RF02: speech-to-text, text analytics, pipeline
├── video_analysis/       # RF01: pose (MediaPipe), objetos (YOLOv8), pipeline, _runner
├── integration/          # fusão, hipóteses, alertas, orquestrador
├── api/
│   ├── main.py           # FastAPI: monitor, demo, intake, patients, health
│   └── static/           # dashboard.html · patients.html · intake.html
├── demo.py               # cenário de demonstração ponta-a-ponta
└── cli.py                # CLI `mmonitor`
scripts/                  # geração de coorte/áudio/pose, demo, dados sintéticos
tests/                    # pytest (41 casos)
docs/                     # relatório técnico, mapa de implementação, checklist, roteiro, deploy
data/
├── samples/              # dados sintéticos de 1 paciente (versionados)
├── patients.json         # coorte de 20 pacientes em um único arquivo
└── patients_media/       # binários da coorte: <ID>.wav (áudio) · <ID>.mp4 (vídeo)
Dockerfile                # imagem para Azure Container Apps
```

## 🧪 Desenvolvimento

```bash
pytest                                        # testes (41 casos)
ruff check .                                  # lint
mypy src                                      # checagem de tipos
python scripts/generate_synthetic_data.py     # (re)gera data/samples/
python scripts/generate_patient_cohort.py     # (re)gera data/patients.json
```

## 🧠 Como as anomalias são detectadas (RF03)

| Método | O que captura | Exemplo |
| ------ | ------------- | ------- |
| **Regras fisiológicas** | Valores fora de faixas clínicas seguras | SpO2 < 88% ⇒ CRÍTICO |
| **Z-score móvel (3.5σ)** | Mudança abrupta vs. linha de base + variação absoluta mínima | FC salta 3.5σ **e** ≥ 15 bpm |
| **IsolationForest** | Combinação multivariada atípica com ao menos 1 sinal fora da faixa | FC alta-normal + SpO2 baixa-normal juntas |
| **Regras de prescrição** | Dose > máx., salto de dose, interações | Warfarina + Ibuprofeno |
| **Movimentação** | Imobilidade prolongada / picos | Queda (pico de movimento) |

### Significância estatística **e** clínica

O z-score e o IsolationForest só reportam anomalia quando o desvio é
estatisticamente raro **e** clinicamente relevante — evita falsos positivos
por ruído. Resultado prático: paciente **estável → score 0.00** (sem alerta);
**crítico → 1.00** (alerta CRITICAL).

### Fusão e `risk_score`

Cada modalidade combina severidade × confiança (peso decrescente `0.5^i`),
**capado pela severidade máxima presente**. A fusão faz a média ponderada
(vitais 1.0, prescrições 0.9, vídeo 0.7, áudio/movimento 0.6) + **bônus de
corroboração** por modalidade adicional em risco relevante (≥0.25), e mapeia o
resultado para severidade e alerta (limiar `0.35`).

## 🔒 Segurança & privacidade

- Credenciais Azure vivem apenas no `.env` (no `.gitignore`), **nunca** no código.
- Pastas `data/raw/` e `data/private/` são ignoradas — **não versione dados
  reais de pacientes**. A coorte em `data/patients.json` é **100% sintética**.
- Vídeos (`*.mp4`) não são versionados por padrão; a coorte usa pose
  pré-extraída e cada paciente pode receber um `video_teste.mp4` local.

## ☁️ Deploy (opcional)

Guia completo em [`docs/deploy_azure.md`](docs/deploy_azure.md) — containeriza a
aplicação (`Dockerfile`) e publica no **Azure Container Apps**, com as chaves
Azure injetadas como *secrets*.

## 📦 Entregáveis da Fase 4

- ✅ **Código-fonte** completo (este repositório) com histórico Git coerente.
- ✅ **Relatório técnico** — [`docs/relatorio_tecnico.md`](docs/relatorio_tecnico.md).
- ✅ **Mapa de implementação** (requisito → código) — [`docs/mapa_implementacao.md`](docs/mapa_implementacao.md).
- ✅ **Checklist de requisitos** — [`docs/checklist_requisitos.md`](docs/checklist_requisitos.md) ([PDF](docs/checklist_requisitos.pdf)).
- ✅ **Roteiro do vídeo** — [`docs/roteiro_video.md`](docs/roteiro_video.md).
- ✅ **Telas web interativas** — servidas pela FastAPI.
- ⬜ **Vídeo demonstrativo** (até 15 min) — a gravar (roteiro pronto).

## 📄 Licença

MIT — ver [`LICENSE`](LICENSE).
