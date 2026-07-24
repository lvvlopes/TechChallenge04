# Dados

Esta pasta armazena datasets locais usados pelo pipeline.

## Estrutura

```
data/
├── samples/          # dados sintéticos de um paciente (gerados por script)
├── patients.json     # coorte de 20 pacientes em UM arquivo (metadados + séries)
├── patients_media/   # binários da coorte: <ID>.wav (áudio) e <ID>.mp4 (vídeo)
├── raw/              # dados brutos externos  — NÃO versionar (no .gitignore)
└── private/          # dados sensíveis/pacientes — NUNCA versionar (no .gitignore)
```

## Coorte de pacientes (`data/patients.json`)

Uma coorte sintética de 20 pacientes num **único arquivo JSON**, cada um com
dados coerentes com seu cenário (estável/crítico), "amarrados" ao ID para
análise automática pela tela `/patients` (ou `POST /api/patients/{id}/analyze`).

```bash
python scripts/generate_patient_cohort.py          # gera os 20 pacientes
python scripts/generate_patient_cohort.py --n 30   # opcional: outra quantidade
```

Cada paciente no JSON contém: metadados (nome, leito, idade, cenário, queixa),
**sinais vitais** e **sinais de pose** em formato colunar, **prescrições** e a
**transcrição** da consulta (fonte do texto quando não há Azure).

Os binários não cabem em JSON e ficam numa pasta plana `data/patients_media/`:

| Arquivo                | Conteúdo                                            |
| ---------------------- | --------------------------------------------------- |
| `<ID>.wav`             | Áudio da consulta (TTS pt-BR, WAV 16 kHz mono)      |
| `<ID>.mp4` *(opcional)*| Vídeo real do paciente                              |

**Vídeo real:** copie um `.mp4` para `data/patients_media/<ID>.mp4` e rode
`python scripts/extract_patient_pose.py` — a pose é extraída do vídeo real e
gravada no `patients.json`; a tela `/patients` passa a exibir o player desse
vídeo. Os `.mp4` **não são versionados** (arquivos grandes, opt-in); os `.wav`
são versionados (gerados por SAPI, Windows-only).

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
