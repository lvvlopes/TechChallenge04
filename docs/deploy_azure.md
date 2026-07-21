# Deploy no Azure — Container Apps

Publicação da API FastAPI de monitoramento multimodal em **Azure Container Apps**,
com a imagem construída pelo **Azure Container Registry** (ACR) a partir deste
repositório.

## Por que Container Apps (e não Vercel)

A aplicação não é compatível com plataformas serverless como o Vercel:

| Requisito da aplicação | Impedimento no serverless |
|---|---|
| `ffmpeg` via `subprocess` (`audio_analysis/speech_to_text.py`) | binário de sistema ausente no runtime Python |
| Uploads gravados em disco (`api/main.py::_save_upload`) | filesystem read-only e efêmero |
| `opencv` + `mediapipe` + `ultralytics` (RF01) | excede o limite de bundle (250 MB) |
| Processamento de vídeo | excede o timeout de função |

O contêiner resolve todos: `ffmpeg` via `apt-get`, disco gravável, sem limite de
bundle e processo de longa duração.

## Modos de build

O [`Dockerfile`](../Dockerfile) tem um build-arg que controla o peso da imagem:

- **Leve (padrão)** — núcleo + Azure Speech/Language. Atende RF02 e RF03; a
  análise de vídeo (RF01) cai no fallback mock. Imagem ~700 MB.
- **Completo** — `--build-arg INSTALL_VISION=true` adiciona OpenCV, MediaPipe e
  YOLO. Imagem ~3 GB, exige contêiner com mais memória.

## Deploy via Azure Cloud Shell

Executar em <https://shell.azure.com> (bash). Dispensa Azure CLI e Docker locais.

```bash
# --- Parâmetros ---
RG=rg-tc4-multimodal
LOC=brazilsouth
ACR=acrtc4multimodal          # precisa ser único globalmente
APP=ca-tc4-multimodal
ENVNAME=cae-tc4-multimodal
REPO=https://github.com/lvvlopes/TechChallenge04.git

# --- Provedores ---
az provider register -n Microsoft.App --wait
az provider register -n Microsoft.ContainerRegistry --wait
az provider register -n Microsoft.OperationalInsights --wait

# --- Grupo e registry ---
az group create -n $RG -l $LOC
az acr create -n $ACR -g $RG --sku Basic --admin-enabled true

# --- Build da imagem a partir do GitHub (roda no Azure) ---
az acr build --registry $ACR --image mmonitor:v1 $REPO

# --- Ambiente e aplicação ---
az extension add -n containerapp --upgrade
az containerapp env create -n $ENVNAME -g $RG -l $LOC

az containerapp create \
  -n $APP -g $RG \
  --environment $ENVNAME \
  --image $ACR.azurecr.io/mmonitor:v1 \
  --registry-server $ACR.azurecr.io \
  --target-port 8000 \
  --ingress external \
  --cpu 1 --memory 2Gi \
  --min-replicas 1 --max-replicas 1 \
  --query properties.configuration.ingress.fqdn -o tsv
```

O último comando imprime a URL pública da aplicação.

> `--min-replicas 1` evita cold start durante a apresentação. Para economizar
> crédito fora do período de demonstração, use `0`.

## Credenciais do Azure Cognitive Services

As chaves **nunca** vão na imagem. Configure-as como variáveis de ambiente:

```bash
az containerapp update -n $APP -g $RG \
  --set-env-vars \
    APP_ENV=production \
    AZURE_SPEECH_KEY=secretref:speech-key \
    AZURE_SPEECH_REGION=brazilsouth \
    AZURE_LANGUAGE_KEY=secretref:language-key \
    AZURE_LANGUAGE_ENDPOINT=https://<recurso>.cognitiveservices.azure.com/
```

Registrando os segredos antes:

```bash
az containerapp secret set -n $APP -g $RG \
  --secrets speech-key=<CHAVE> language-key=<CHAVE>
```

Sem essas variáveis a aplicação sobe normalmente em modo degradado — ver as
propriedades `azure_speech_enabled` / `azure_language_enabled` em
[`config.py`](../src/multimodal_monitor/config.py).

## Verificação

```bash
FQDN=$(az containerapp show -n $APP -g $RG \
  --query properties.configuration.ingress.fqdn -o tsv)

curl https://$FQDN/health      # capacidades ativas
```

Interfaces web: `https://$FQDN/` (dashboard), `https://$FQDN/intake`
(captura clínica) e `https://$FQDN/docs` (OpenAPI).

## Atualizando após mudanças no código

```bash
az acr build --registry $ACR --image mmonitor:v2 $REPO
az containerapp update -n $APP -g $RG --image $ACR.azurecr.io/mmonitor:v2
```

## Removendo tudo

```bash
az group delete -n $RG --yes --no-wait
```
