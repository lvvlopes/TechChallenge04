# =====================================================================
# Gera o áudio (.wav) de cada paciente da coorte a partir da sua
# transcrição (consulta.txt), usando a voz SAPI pt-BR do Windows e
# normalizando para WAV PCM 16 kHz mono — o formato nativo do Azure
# Speech to Text.
#
# Uso:  powershell -ExecutionPolicy Bypass -File scripts\generate_patient_audio.ps1
#
# Requisitos: Windows com voz pt-BR (Microsoft Maria) e ffmpeg no PATH.
# =====================================================================

# Continue: o ffmpeg escreve no stderr normalmente; sob "Stop" o PowerShell
# trataria isso como erro fatal. Verificamos $LASTEXITCODE manualmente.
$ErrorActionPreference = "Continue"
Add-Type -AssemblyName System.Speech

$root = Join-Path (Split-Path $PSScriptRoot -Parent) "data\patients"
if (-not (Test-Path $root)) {
    Write-Error "Coorte não encontrada em $root. Rode antes: python scripts\generate_patient_cohort.py"
    exit 1
}

$ffmpeg = (Get-Command ffmpeg -ErrorAction SilentlyContinue).Source
if (-not $ffmpeg) { Write-Error "ffmpeg não encontrado no PATH."; exit 1 }

$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
# seleciona uma voz pt-BR se existir
$ptVoice = $synth.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture -like "pt*" } | Select-Object -First 1
if ($ptVoice) { $synth.SelectVoice($ptVoice.VoiceInfo.Name); Write-Output ("Voz: " + $ptVoice.VoiceInfo.Name) }
else { Write-Output "Aviso: nenhuma voz pt-BR encontrada; usando a voz padrão." }
$synth.Rate = -1  # levemente mais devagar → melhor reconhecimento

$patients = Get-ChildItem $root -Directory | Where-Object { $_.Name -like "PAC-*" }
$count = 0
foreach ($p in $patients) {
    $txt = Join-Path $p.FullName "consulta.txt"
    if (-not (Test-Path $txt)) { continue }
    $text = Get-Content $txt -Raw -Encoding UTF8
    $rawWav = Join-Path $env:TEMP ($p.Name + "_raw.wav")
    $outWav = Join-Path $p.FullName "consulta.wav"

    # 1) SAPI sintetiza para um WAV temporário
    $synth.SetOutputToWaveFile($rawWav)
    $synth.Speak($text)
    $synth.SetOutputToNull()

    # 2) ffmpeg normaliza para PCM 16 kHz mono (formato do Azure Speech)
    & $ffmpeg -hide_banner -loglevel error -nostats -y -i $rawWav `
        -ar 16000 -ac 1 -c:a pcm_s16le $outWav | Out-Null
    Remove-Item $rawWav -ErrorAction SilentlyContinue

    if (Test-Path $outWav) {
        $count++
        Write-Output ("  " + $p.Name + " -> consulta.wav")
    } else {
        Write-Output ("  " + $p.Name + " -> FALHOU")
    }
}
$synth.Dispose()
Write-Output ("Concluído: " + $count + " áudios gerados em " + $root)
