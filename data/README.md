# Dados

Esta pasta armazena datasets locais usados pelo pipeline.

## Estrutura

```
data/
├── samples/     # dados sintéticos de um paciente (gerados por script)
├── patients/    # coorte de 20 pacientes com cenários estável/crítico
├── raw/         # dados brutos externos  — NÃO versionar (no .gitignore)
└── private/     # dados sensíveis/pacientes — NUNCA versionar (no .gitignore)
```

## Coorte de pacientes (`data/patients/`)

Uma coorte sintética de 20 pacientes, cada um com dados coerentes com seu
cenário (estável ou crítico), "amarrados" ao ID do paciente para análise
automática pela tela `/patients` (ou pela API `POST /api/patients/{id}/analyze`).

```bash
python scripts/generate_patient_cohort.py          # gera os 20 pacientes
python scripts/generate_patient_cohort.py --n 30   # opcional: outra quantidade
```

Estrutura por paciente (`data/patients/PAC-XXX/`):

| Arquivo             | Conteúdo                                              |
| ------------------- | ----------------------------------------------------- |
| `vitals.csv`        | Série temporal de sinais vitais (coerente c/ cenário) |
| `prescriptions.csv` | Evolução de prescrições                               |
| `pose_frames.csv`   | Sinais de pose de vídeo (movimento + tronco)          |
| `consulta.txt`      | Transcrição da consulta (fonte do STT em modo mock)   |

O manifesto `data/patients/cohort.json` lista os pacientes com metadados
(nome, leito, idade, cenário, queixa). Pacientes **críticos** têm o vídeo real
`data/samples/video_teste.mp4` vinculado — a opção "usar vídeo real" na tela
processa esse arquivo com MediaPipe + YOLOv8; caso contrário usa a pose
pré-computada (mais rápido).

## Gerando os dados de exemplo

```bash
python scripts/generate_synthetic_data.py
```

Isso cria em `data/samples/`:

| Arquivo                   | Conteúdo                                            |
| ------------------------- | --------------------------------------------------- |
| `vitals.csv`              | Série temporal de sinais vitais (com anomalias)     |
| `prescriptions.csv`       | Evolução de prescrições (com anomalias)             |
| `pose_frames.csv`         | Sinais posturais de vídeo (movimento + tronco)      |
| `consulta_critica.txt`    | Transcrição de referência (paciente em risco)       |
| `consulta_estavel.txt`    | Transcrição de referência (paciente estável)        |

## Datasets reais sugeridos

- **PhysioNet** — https://physionet.org/ (sinais vitais / séries temporais)
- **Google AudioSet** — https://research.google.com/audioset/ (áudio)

> ⚠️ Nunca faça commit de dados reais de pacientes. Use apenas dados públicos
> ou sintéticos. As pastas `raw/` e `private/` estão no `.gitignore`.
