# =====================================================================
# Gera o áudio (.wav) de cada paciente da coorte a partir da transcrição
# gravada em data/patients.json, usando a voz SAPI pt-BR do Windows e
# normalizando para WAV PCM 16 kHz mono — o formato nativo do Azure
# Speech to Text.
#
# Saída: data/patients_media/<ID>.wav
#
# Uso:  powershell -ExecutionPolicy Bypass -File scripts\generate_patient_audio.ps1
#
# Requisitos: Windows com voz pt-BR (Microsoft Maria) e ffmpeg no PATH.
# =====================================================================

# Continue: o ffmpeg escreve no stderr normalmente; sob "Stop" o PowerShell
# trataria isso como erro fatal. Verificamos a saída manualmente.
$ErrorActionPreference = "Continue"
Add-Type -AssemblyName System.Speech

$root      = Split-Path $PSScriptRoot -Parent
$cohortFile = Join-Path $root "data\patients.json"
$mediaDir   = Join-Path $root "data\patients_media"

if (-not (Test-Path $cohortFile)) {
    Write-Error "Coorte não encontrada em $cohortFile. Rode antes: python scripts\generate_patient_cohort.py"
    exit 1
}
if (-not (Test-Path $mediaDir)) { New-Item -ItemType Directory -Path $mediaDir | Out-Null }

$ffmpeg = (Get-Command ffmpeg -ErrorAction SilentlyContinue).Source
if (-not $ffmpeg) { Write-Error "ffmpeg não encontrado no PATH."; exit 1 }

$cohort = Get-Content $cohortFile -Raw -Encoding UTF8 | ConvertFrom-Json

$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$ptVoice = $synth.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture -like "pt*" } | Select-Object -First 1
if ($ptVoice) { $synth.SelectVoice($ptVoice.VoiceInfo.Name); Write-Output ("Voz: " + $ptVoice.VoiceInfo.Name) }
else { Write-Output "Aviso: nenhuma voz pt-BR encontrada; usando a voz padrão." }
$synth.Rate = -1  # levemente mais devagar → melhor reconhecimento

$count = 0
foreach ($p in $cohort.patients) {
    if (-not $p.transcript) { continue }
    $rawWav = Join-Path $env:TEMP ($p.id + "_raw.wav")
    $outWav = Join-Path $mediaDir ($p.id + ".wav")

    # 1) SAPI sintetiza para um WAV temporário
    $synth.SetOutputToWaveFile($rawWav)
    $synth.Speak($p.transcript)
    $synth.SetOutputToNull()

    # 2) ffmpeg normaliza para PCM 16 kHz mono (formato do Azure Speech)
    & $ffmpeg -hide_banner -loglevel error -nostats -y -i $rawWav `
        -ar 16000 -ac 1 -c:a pcm_s16le $outWav | Out-Null
    Remove-Item $rawWav -ErrorAction SilentlyContinue

    if (Test-Path $outWav) {
        $count++
        Write-Output ("  " + $p.id + " -> " + $p.id + ".wav")
    } else {
        Write-Output ("  " + $p.id + " -> FALHOU")
    }
}
$synth.Dispose()
Write-Output ("Concluído: " + $count + " áudios gerados em " + $mediaDir)
