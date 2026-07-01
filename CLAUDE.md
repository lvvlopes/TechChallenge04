# Tech Challenge Fase 4 — Monitoramento Multimodal de Pacientes

## Visão Geral
Sistema de IA para monitoramento contínuo de pacientes hospitalares por meio de
dados multimodais (áudio, vídeo e texto), com objetivo de identificar sinais
precoces de risco clínico. O sistema deve analisar vídeos (cirurgias/fisioterapia),
processar áudio de consultas (fala/sintomas vocais) e detectar anomalias em
sinais vitais, prescrições e evolução clínica, alertando a equipe médica em
tempo real.

Este projeto é o Tech Challenge da Fase 4 do curso FIAP Pós-Tech em IA (POSTECH
IADT), valendo 90% da nota das disciplinas da fase.

## Contexto do Desafio
Dando continuidade a um sistema hospitalar que já usa IA para analisar exames,
documentos e apoiar decisões clínicas, o hospital quer evoluir para
monitoramento contínuo e multimodal dos pacientes, cruzando:
- Vídeo (cirurgias, sessões de fisioterapia)
- Áudio (consultas médicas, fala do paciente)
- Texto (laudos, prescrições, sinais vitais registrados)

## Objetivos do Projeto
- Realizar análise e fusão de diferentes tipos de dados médicos (texto, áudio, vídeo).
- Utilizar serviços em nuvem (Azure Cognitive Services) para ampliar capacidade
  de processamento e inteligência.
- Aplicar técnicas de detecção de anomalias em tempo real para monitoramento
  preventivo.

## Stack Tecnológica
> Preencher/confirmar conforme decisões de arquitetura. Sugestão baseada no
> desafio e em experiência prévia (MediaPipe/OpenCV já usados na Fase 4 anterior):
- Linguagem principal: Python
- Visão computacional: OpenCV, MediaPipe, OpenPose, YOLOv8
- Áudio: Azure Speech to Text, Azure Text Analytics
- Séries temporais / anomalias: bibliotecas de detecção de anomalias (ex.
  `pyod`, `statsmodels`, `scikit-learn` — a definir)
- Cloud: Azure Cognitive Services
- Orquestração/API (se aplicável): FastAPI
- Versionamento: Git / GitHub

## Requisitos Funcionais

### RF01 — Análise de Vídeo
- Processar vídeos clínicos (ex.: sessões de fisioterapia ou cirurgias gravadas).
- Detectar movimentos ou eventos fora do padrão esperado usando:
  - OpenPose para análise postural;
  - YOLOv8 para detecção de objetos e áreas críticas.
- Gerar relatórios automáticos indicando desvios ou falhas no procedimento.

### RF02 — Análise de Áudio
- Processar áudios de consultas médicas.
- Detectar alterações vocais indicativas de condições médicas (ex.: cansaço,
  dificuldades respiratórias, fadiga, disartria).
- Utilizar Azure Speech to Text para transcrever e analisar os áudios.
- Identificar termos críticos e sentimentos com Azure Text Analytics.

### RF03 — Detecção de Anomalias
Aplicar técnicas de detecção de anomalias em:
- Séries temporais de sinais vitais (batimentos, pressão arterial, oxigenação);
- Evolução de prescrições (alterações inesperadas no tratamento);
- Padrões de movimentação do paciente durante a internação.

Gerar alertas automáticos para a equipe médica com base nas anomalias detectadas.

## Requisitos Não-Funcionais
- Processamento com suporte a fusão multimodal (texto + áudio + vídeo).
- Integração com serviços gerenciados em nuvem (Azure Cognitive Services).
- Capacidade de gerar alertas em tempo real (ou near real-time) para a equipe médica.
- Código organizado em repositório Git, com histórico de commits coerente.

## Datasets Sugeridos
- PhysioNet — https://physionet.org/ (sinais vitais / séries temporais clínicas)
- Google AudioSet — https://research.google.com/audioset/ (dados de áudio)

## Entregáveis da Fase 4
1. **Repositório Git** contendo:
   - Código-fonte completo da solução;
   - Relatório técnico com:
     - Descrição do fluxo multimodal;
     - Modelos aplicados em cada tipo de dado;
     - Resultados obtidos e exemplos de anomalias detectadas.
2. **Vídeo demonstrativo** (até 15 minutos):
   - Upload no YouTube ou Vimeo (público ou não listado);
   - Deve demonstrar:
     - Exemplo prático da análise de áudio e vídeo;
     - Detecção e resposta a anomalias;
     - Integração dos serviços Azure;
     - Fluxo final do alerta à equipe médica.

## Estrutura do Projeto (sugestão inicial)
```
/src
  /video_analysis      # OpenPose, YOLOv8, processamento de vídeo
  /audio_analysis       # Azure Speech to Text, Text Analytics
  /anomaly_detection     # séries temporais, prescrições, movimentação
  /integration           # orquestração multimodal + alertas
/data                    # datasets locais (não versionar dados sensíveis)
/notebooks                # exploração e prototipagem
/docs
  relatorio_tecnico.md
/tests
CLAUDE.md
README.md
requirements.txt
```
*(ajustar conforme decisões reais de arquitetura)*

## Convenções de Código
- A definir — ex.: PEP8, type hints, docstrings, nomes de funções em inglês/português (definir padrão).

## Fora de Escopo
- Não definido explicitamente no enunciado — validar com o grupo se algo deve
  ser excluído (ex.: dados reais de pacientes, apenas dados públicos/sintéticos).

## Observações para o Claude Code
- Este é um projeto acadêmico (FIAP Pós-Tech) com entrega obrigatória de
  repositório Git + relatório técnico + vídeo demonstrativo.
- Priorizar clareza e reprodutibilidade do pipeline multimodal ao gerar código.
- Ao sugerir integrações Azure, considerar que credenciais/chaves de API não
  devem ser commitadas (usar `.env` + `.gitignore`).
