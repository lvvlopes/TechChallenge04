# Dados

Esta pasta armazena datasets locais usados pelo pipeline.

## Estrutura

```
data/
├── samples/     # dados sintéticos versionáveis (gerados por script)
├── raw/         # dados brutos externos  — NÃO versionar (no .gitignore)
└── private/     # dados sensíveis/pacientes — NUNCA versionar (no .gitignore)
```

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
