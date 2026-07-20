"""Geração de hipóteses interpretativas a partir dos achados multimodais.

IMPORTANTE — escopo clínico: este módulo NÃO produz diagnóstico. Ele mapeia
padrões observados nos achados para *hipóteses de causa possível*, servindo
como apoio à decisão da equipe médica (que permanece responsável por toda
conclusão clínica). Todas as hipóteses são acompanhadas do disclaimer em
:data:`DISCLAIMER`.

As regras são deliberadamente conservadoras e interpretáveis: cada uma
verifica um padrão explícito (via ``metadata`` dos achados) e emite um texto
com o formato "observação — compatível com: [causas]".
"""

from __future__ import annotations

from ..schemas import Finding, Modality

DISCLAIMER = (
    "Hipóteses geradas automaticamente por regras de apoio à decisão — "
    "não substituem avaliação médica."
)

# Termos de fala/sintomas agrupados por padrão clínico.
_DYSPNEA_TERMS = {
    "falta de ar",
    "dificuldade para respirar",
    "nao consigo respirar",
}
_CHEST_PAIN_TERMS = {"dor no peito", "aperto no peito"}
_NEURO_TERMS = {"desmaio", "desmaiei", "tontura", "confusao", "dormencia"}
_FATIGUE_TERMS = {"cansaco", "cansada", "cansado", "fadiga", "fraqueza"}

# Nº máximo de hipóteses no alerta (as mais específicas primeiro).
MAX_HYPOTHESES = 5


def generate_hypotheses(findings: list[Finding]) -> list[str]:
    """Deriva hipóteses interpretativas a partir dos achados.

    Retorna uma lista ordenada (mais específicas/corroboradas primeiro),
    limitada a :data:`MAX_HYPOTHESES`. Lista vazia quando nenhum padrão
    conhecido é reconhecido.
    """
    flags = _extract_flags(findings)
    hypotheses: list[str] = []

    # --- Combos multimodais (mais específicos — entram primeiro) ---------- #
    if flags["dyspnea_reported"] and flags["desaturation"]:
        hypotheses.append(
            "Dispneia relatada corroborada por dessaturação objetiva — "
            "priorizar avaliação respiratória (compatível com: insuficiência "
            "respiratória aguda, broncoaspiração, tromboembolismo pulmonar)"
        )
    if flags["chest_pain"] and (flags["tachycardia"] or flags["hypertension"]):
        hypotheses.append(
            "Dor torácica associada a alteração hemodinâmica — avaliar com "
            "urgência (compatível com: síndrome coronariana aguda, crise "
            "hipertensiva, dissecção)"
        )
    if flags["movement_spike"] and flags["horizontal_posture"]:
        hypotheses.append(
            "Pico de movimento seguido de padrão horizontal sustentado — "
            "compatível com: queda não assistida (verificar paciente "
            "imediatamente)"
        )

    # --- Padrões de modalidade única -------------------------------------- #
    if flags["horizontal_posture"] and not flags["movement_spike"]:
        hypotheses.append(
            "Padrão postural horizontal sustentado — compatível com: paciente "
            "acamado/repouso, queda não assistida, incapacidade de "
            "reposicionamento"
        )
    if flags["movement_spike"] and not flags["horizontal_posture"]:
        hypotheses.append(
            "Pico abrupto de movimentação — compatível com: queda, crise "
            "convulsiva, agitação psicomotora"
        )
    if flags["immobility"]:
        hypotheses.append(
            "Imobilidade prolongada — compatível com: sedação excessiva, "
            "rebaixamento de consciência; risco de lesão por pressão e "
            "tromboembolismo venoso"
        )
    if flags["desaturation"] and not flags["dyspnea_reported"]:
        hypotheses.append(
            "Dessaturação de oxigênio — compatível com: insuficiência "
            "respiratória, broncoaspiração, atelectasia"
        )
    if flags["chest_pain"] and not (flags["tachycardia"] or flags["hypertension"]):
        hypotheses.append(
            "Queixa de dor torácica — compatível com: síndrome coronariana, "
            "causa musculoesquelética, refluxo (investigar)"
        )
    if flags["dyspnea_reported"] and not flags["desaturation"]:
        hypotheses.append(
            "Dispneia relatada sem dessaturação registrada — corroborar com "
            "oximetria e frequência respiratória"
        )
    if flags["tachycardia"] and not flags["chest_pain"]:
        hypotheses.append(
            "Taquicardia — compatível com: dor não controlada, febre, "
            "hipovolemia, resposta adrenérgica"
        )
    if flags["bradycardia"]:
        hypotheses.append(
            "Bradicardia — compatível com: efeito medicamentoso "
            "(betabloqueador/digitálico), distúrbio de condução"
        )
    if flags["hypertension"] and not flags["chest_pain"]:
        hypotheses.append(
            "Elevação pressórica — compatível com: crise hipertensiva, dor "
            "não controlada, ansiedade"
        )
    if flags["fever"]:
        hypotheses.append(
            "Hipertermia — compatível com: processo infeccioso, reação "
            "medicamentosa"
        )
    if flags["neuro_symptoms"]:
        hypotheses.append(
            "Sintomas neurológicos relatados (tontura/desmaio/confusão/"
            "dormência) — compatível com: evento neurológico agudo, "
            "hipotensão, hipoglicemia"
        )
    if flags["fatigue_reported"] and flags["multivariate_anomaly"]:
        hypotheses.append(
            "Fadiga relatada com padrão multivariado atípico de sinais "
            "vitais — avaliar descompensação subclínica"
        )
    if flags["drug_interaction"]:
        hypotheses.append(
            "Interação medicamentosa conhecida entre prescrições ativas — "
            "revisar farmacoterapia com a farmácia clínica"
        )
    if flags["dose_anomaly"]:
        hypotheses.append(
            "Alteração posológica atípica (salto de dose ou dose acima da "
            "referência) — confirmar intenção do prescritor"
        )

    return hypotheses[:MAX_HYPOTHESES]


# --------------------------------------------------------------------------- #
def _extract_flags(findings: list[Finding]) -> dict[str, bool]:
    """Varre os achados e liga flags de padrões clínicos reconhecidos."""
    flags = {
        "horizontal_posture": False,
        "movement_spike": False,
        "immobility": False,
        "desaturation": False,
        "tachycardia": False,
        "bradycardia": False,
        "hypertension": False,
        "fever": False,
        "multivariate_anomaly": False,
        "dyspnea_reported": False,
        "chest_pain": False,
        "neuro_symptoms": False,
        "fatigue_reported": False,
        "drug_interaction": False,
        "dose_anomaly": False,
    }

    for f in findings:
        meta = f.metadata
        rule = meta.get("rule")
        signal = meta.get("signal")
        term = meta.get("term", "")
        term_norm = _strip_accents(str(term).lower())

        # --- vídeo / movimentação ---
        trunk = meta.get("trunk_angle")
        if trunk is not None and float(trunk) >= 80:
            flags["horizontal_posture"] = True
        if rule == "spike":
            flags["movement_spike"] = True
        if rule == "immobility":
            flags["immobility"] = True

        # --- sinais vitais ---
        value = meta.get("value")
        zscore = meta.get("zscore", 0.0)
        if signal == "spo2":
            low_value = value is not None and float(value) < 92
            if low_value or (zscore is not None and float(zscore) < 0):
                flags["desaturation"] = True
        elif signal == "heart_rate":
            if value is not None:
                if float(value) > 100:
                    flags["tachycardia"] = True
                elif float(value) < 50:
                    flags["bradycardia"] = True
            elif zscore and float(zscore) > 0:
                flags["tachycardia"] = True
        elif signal in ("systolic_bp", "diastolic_bp"):
            if (value is not None and _is_high_bp(signal, float(value))) or (
                value is None and zscore and float(zscore) > 0
            ):
                flags["hypertension"] = True
        elif signal == "temperature":
            if value is not None and float(value) > 37.8:
                flags["fever"] = True
        if meta.get("method") == "isolation_forest":
            flags["multivariate_anomaly"] = True

        # --- fala / sintomas (modalidade AUDIO) ---
        if f.modality is Modality.AUDIO and term_norm:
            if term_norm in {_strip_accents(t) for t in _DYSPNEA_TERMS}:
                flags["dyspnea_reported"] = True
            if term_norm in {_strip_accents(t) for t in _CHEST_PAIN_TERMS}:
                flags["chest_pain"] = True
            if term_norm in {_strip_accents(t) for t in _NEURO_TERMS}:
                flags["neuro_symptoms"] = True
            if term_norm in {_strip_accents(t) for t in _FATIGUE_TERMS}:
                flags["fatigue_reported"] = True

        # --- prescrições ---
        if rule == "interaction":
            flags["drug_interaction"] = True
        if rule in ("dose_jump", "max_dose"):
            flags["dose_anomaly"] = True

    return flags


def _is_high_bp(signal: str, value: float) -> bool:
    return value > 140 if signal == "systolic_bp" else value > 90


def _strip_accents(text: str) -> str:
    import unicodedata

    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))
