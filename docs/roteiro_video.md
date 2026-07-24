# Roteiro do Vídeo Demonstrativo — Tech Challenge Fase 4

**Duração-alvo:** 10–12 min (limite: 15 min) · **Formato:** gravação de tela + narração

Este roteiro está organizado para **mostrar cada requisito do enunciado** de
forma visível, com o que falar, o que aparecer na tela e o comando a executar.
Prepare o ambiente antes de gravar (ver "Preparação" no fim).

Legenda: 🎙️ = fala/narração · 🖥️ = tela · ⌨️ = comando a executar

---

## Bloco 0 — Abertura (0:00 – 1:00)

🎙️ *"Olá, somos o grupo [nomes] do curso FIAP Pós-Tech IADT. Este é o Tech
Challenge da Fase 4: um sistema de IA para monitoramento contínuo e multimodal
de pacientes hospitalares, que cruza vídeo, áudio e texto para detectar sinais
precoces de risco e alertar a equipe médica em tempo real."*

🖥️ Mostrar o `README.md` aberto (a tabela dos 3 requisitos e o diagrama de
arquitetura).

🎙️ *"A arquitetura tem três pipelines — um por modalidade — que produzem
achados padronizados. Uma camada de fusão multimodal combina esses achados num
score de risco e dispara o alerta clínico."*

> 💡 **Dica de demonstração:** a tela **`/patients`** (coorte de 20 pacientes)
> é o caminho mais forte para o vídeo — cada paciente já tem áudio, vídeo,
> vitais e prescrições "amarrados" ao ID. Selecionar o **PAC-001** (crítico,
> com vídeo real 🎥) demonstra as 3 modalidades + fusão + alerta numa única
> ação; um paciente **estável** ao lado mostra o contraste (score 0.00). Os
> comandos de terminal abaixo continuam úteis para "mostrar o código rodando".

---

## Bloco 1 — Análise de Vídeo / RF01 (1:00 – 4:00)

### 1.1 + 1.2 Pose (postura) + YOLOv8 (objetos)

🎙️ *"Requisito 1: análise de vídeo. Vamos processar um vídeo clínico real.
Para análise postural usamos o MediaPipe Pose — que extrai 33 pontos do corpo,
mesma finalidade do OpenPose sugerido no enunciado, mas rodando em CPU. E o
YOLOv8 para detecção de objetos e áreas críticas."*

⌨️ Terminal:
```bash
python -c "from multimodal_monitor.video_analysis import VideoAnalysisPipeline; r = VideoAnalysisPipeline().process('data/samples/video_teste.mp4'); print(r.summary); [print(' -', f.severity.value, '|', f.description[:90]) for f in r.findings[:5]]"
```

🖥️ Apontar na saída: *"Pose extraída de N quadros"*, *"Detecções de objetos: N"*,
e os achados de desvio postural.

🎙️ *"O sistema extraiu a pose de cada quadro, rodou o YOLOv8 para detectar
objetos, e identificou desvios posturais — aqui, inclinação de tronco fora do
padrão esperado."*

### (Opcional, muito visual) YOLOv8 com bounding boxes

⌨️
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').predict('data/samples/video_teste.mp4', save=True, project='outputs', name='yolo')"
```
🖥️ Abrir `outputs/yolo/video_teste.mp4` — mostrar as caixas desenhadas nas pessoas.

### 1.3 Relatório automático

🎙️ *"Cada análise gera um relatório estruturado — os achados com severidade,
descrição, score e momento. É isso que alimenta a camada de decisão."*

🖥️ Mostrar rapidamente [`video_pipeline.py:58`](../src/multimodal_monitor/video_analysis/video_pipeline.py)
(`_build_result` retornando o `ModalityResult`).

---

## Bloco 2 — Análise de Áudio / RF02 (4:00 – 7:00)

### 2.3 + 2.4 Azure Speech to Text + Text Analytics

🎙️ *"Requisito 2: análise de áudio. Primeiro, confirmo que os serviços Azure
estão realmente ativos."*

🖥️ Abrir <http://127.0.0.1:8000/health> no navegador (ou `curl`):
```bash
curl http://127.0.0.1:8000/health
```
🖥️ Destacar: `"azure_speech": true`, `"azure_language": true`.

🎙️ *"Ambos os serviços Azure estão conectados. Agora vou até a tela de captura
clínica e gravo a fala de um paciente."*

🖥️ Abrir <http://127.0.0.1:8000/intake>. Na seção **Áudio** → aba **Gravar do
microfone** → gravar falando claramente:
> *"Doutor, estou com muita falta de ar e uma dor no peito que não passa, me
> sinto muito cansado."*

### 2.1 + 2.2 Processar áudio + detectar alterações vocais

🎙️ *"Além do áudio, preencho alguns sinais vitais alterados e uma prescrição,
para o caso completo."*

🖥️ Na mesma tela: SpO₂ = **84**, FC = **130**; adicionar prescrição
`warfarina 5mg` e `ibuprofeno 600mg`. Clicar **Processar admissão**.

🖥️ No painel de resultado, apontar:
- Modalidade **Áudio/sintomas** com os termos críticos detectados
- O painel de transcrição (a fala foi transcrita pelo **Azure Speech**)
- O sentimento negativo identificado (**Azure Text Analytics**)

🎙️ *"O Azure transcreveu minha fala, o Text Analytics identificou sentimento
negativo, e o sistema extraiu os termos críticos: 'falta de ar', 'dor no peito',
'cansaço' — cada um com sua severidade clínica."*

---

## Bloco 3 — Detecção de Anomalias / RF03 (7:00 – 10:00)

### 3.1 As três fontes de anomalia

🎙️ *"Requisito 3: detecção de anomalias. Uso o dashboard de monitoramento, que
roda o pipeline sobre uma série temporal completa."*

🖥️ Abrir <http://127.0.0.1:8000/> (dashboard). Deixar no **cenário crítico** e
clicar **Executar ciclo**.

🖥️ Apontar cada parte:
- **Gráficos de sinais vitais** — mostrar a queda de SpO₂ e o pico de FC
- **Feed de achados** — filtrar por severidade "crítico"

🎙️ *"Nos sinais vitais, o sistema usa três técnicas combinadas: regras
fisiológicas, z-score para mudanças abruptas, e IsolationForest para padrões
multivariados atípicos. Ele detectou a dessaturação de oxigênio, a taquicardia
e o pico hipertensivo."*

🎙️ *"Também detecta anomalias em prescrições — aqui, um salto de dose e uma
interação medicamentosa entre warfarina e ibuprofeno — e em padrões de
movimentação, como imobilidade prolongada ou quedas."*

### Contraste estável × crítico

🖥️ Alternar para **cenário estável** e clicar **Executar ciclo**. Mostrar o
score cair e o alerta sumir/baixar.

🎙️ *"No cenário estável, o mesmo pipeline não gera alerta relevante — provando
que a detecção é seletiva, não dispara para tudo."*

---

## Bloco 4 — Fusão + Alerta à equipe médica / RF3.2 (10:00 – 11:30)

🎙️ *"Por fim, o requisito do alerta. A fusão multimodal combina todas as
modalidades. Quando várias concordam num risco, a confiança aumenta — é o efeito
de corroboração."*

🖥️ Voltar ao **cenário crítico** (ou ao resultado do `/intake`). Mostrar o
**banner de alerta CRÍTICO** no topo, com:
- Título e mensagem
- As **hipóteses de apoio à decisão** (ex.: "dispneia relatada corroborada por
  dessaturação objetiva")
- O aviso "não substituem avaliação médica"

🎙️ *"O sistema gera o alerta com score 1.0, severidade crítica, listando os
achados que mais contribuíram e hipóteses interpretativas para apoiar — mas não
substituir — a decisão do médico."*

🖥️ (Opcional) Mostrar `ALERT_CHANNEL=webhook` e o alerta chegando num canal do
Microsoft Teams/Slack.

🎙️ *"O alerta pode ser despachado para o console, um arquivo de auditoria, ou
diretamente para o Teams/Slack da equipe médica."*

---

## Bloco 5 — Encerramento (11:30 – 12:00)

🎙️ *"Recapitulando: análise de vídeo com MediaPipe e YOLOv8, análise de áudio
com Azure Speech e Text Analytics, detecção de anomalias em sinais vitais,
prescrições e movimentação, tudo fundido num alerta multimodal em tempo real.
O código está no repositório, com 36 testes automatizados e o relatório técnico
completo. Obrigado!"*

🖥️ Mostrar rapidamente: `pytest` passando (36 testes verdes) e a estrutura de
pastas do projeto.

⌨️ (se quiser mostrar ao vivo)
```bash
pytest -q
```

---

## Preparação (antes de gravar)

1. **Ambiente Azure ativo** — `.env` preenchido com as chaves do Speech e do
   Language. Confirmar com `curl http://127.0.0.1:8000/health`.
2. **Servidor no ar:**
   ```bash
   uvicorn multimodal_monitor.api.main:app --reload --app-dir src
   ```
3. **Coorte gerada** — `python scripts/generate_patient_cohort.py`, áudios com
   `powershell -File scripts/generate_patient_audio.ps1`, e o vídeo do PAC-001:
   ```bash
   cp data/samples/video_teste.mp4 data/patients_media/PAC-001.mp4
   python scripts/extract_patient_pose.py
   ```
4. **Vídeo de teste** em `data/samples/video_teste.mp4`.
5. **Microfone e webcam** testados e com permissão concedida no navegador.
6. **Terminal com fonte grande** (para legibilidade na gravação).
7. **Fechar abas/notificações** que possam poluir a tela.
8. **Ensaio do áudio:** treine a fala do paciente uma vez para sair natural.

## Mapa requisito → bloco do vídeo

| Requisito | Bloco | Momento |
|---|---|---|
| 1.1 Processar vídeos | 1 | ~1:00 |
| 1.2 Pose (OpenPose/MediaPipe) | 1 | ~1:30 |
| 1.2 YOLOv8 | 1 | ~2:30 |
| 1.3 Relatórios automáticos | 1 | ~3:30 |
| 2.1 Processar áudios | 2 | ~5:30 |
| 2.2 Alterações vocais | 2 | ~6:00 |
| 2.3 Azure Speech to Text | 2 | ~4:30 |
| 2.4 Azure Text Analytics | 2 | ~6:30 |
| 3.1 Anomalias — vitais | 3 | ~7:30 |
| 3.1 Anomalias — prescrições | 3 | ~8:30 |
| 3.1 Anomalias — movimentação | 3 | ~9:00 |
| 3.2 Alertas automáticos | 4 | ~10:00 |
