"""Detecção de anomalias em séries temporais de sinais vitais (RF03).

Combina três abordagens complementares:

1. **Regras fisiológicas** — faixas clínicas de segurança (ex.: SpO2 < 90%).
   Interpretáveis e sempre ativas; capturam eventos críticos óbvios.
2. **Z-score móvel** — desvio estatístico em relação a uma janela recente,
   detectando mudanças abruptas relativas à linha de base do paciente.
3. **IsolationForest** (multivariado) — captura combinações anômalas de
   sinais que, isoladamente, pareceriam normais.

O resultado é uma lista de :class:`Finding` normalizada para a camada de fusão.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..logging_config import get_logger
from ..schemas import Finding, Modality, Severity

logger = get_logger(__name__)


# Faixas fisiológicas de referência para adultos.
# (limite_baixo_crítico, baixo, alto, alto_crítico)
PHYSIOLOGICAL_RANGES: dict[str, tuple[float, float, float, float]] = {
    "heart_rate": (40, 50, 100, 130),
    "systolic_bp": (80, 90, 140, 180),
    "diastolic_bp": (50, 60, 90, 120),
    "spo2": (88, 92, 100, 100),
    "respiratory_rate": (8, 12, 20, 28),
    "temperature": (35.0, 36.0, 37.8, 39.0),
}

# Rótulos legíveis para mensagens de alerta.
_LABELS = {
    "heart_rate": "frequência cardíaca",
    "systolic_bp": "pressão sistólica",
    "diastolic_bp": "pressão diastólica",
    "spo2": "saturação de oxigênio (SpO2)",
    "respiratory_rate": "frequência respiratória",
    "temperature": "temperatura",
}


@dataclass
class VitalsAnomalyDetector:
    """Detector de anomalias em sinais vitais.

    Parameters
    ----------
    contamination:
        Fração esperada de amostras anômalas para o IsolationForest.
    zscore_threshold:
        Limiar (em desvios-padrão) para o detector de z-score móvel.
    window:
        Tamanho da janela móvel (nº de leituras) para a linha de base.
    """

    contamination: float = 0.02
    zscore_threshold: float = 3.0
    window: int = 20
    _feature_columns: list[str] = field(default_factory=list, init=False)

    # ------------------------------------------------------------------ #
    # API pública
    # ------------------------------------------------------------------ #
    def detect(self, df: pd.DataFrame) -> list[Finding]:
        """Analisa um DataFrame de sinais vitais e retorna achados.

        O DataFrame deve conter uma coluna ``timestamp`` e um subconjunto das
        colunas em :data:`PHYSIOLOGICAL_RANGES`.
        """
        if df.empty:
            return []

        df = df.sort_values("timestamp").reset_index(drop=True)
        self._feature_columns = [c for c in PHYSIOLOGICAL_RANGES if c in df.columns]

        findings: list[Finding] = []
        findings.extend(self._rule_based(df))
        findings.extend(self._zscore(df))
        findings.extend(self._isolation_forest(df))
        return findings

    # ------------------------------------------------------------------ #
    # 1. Regras fisiológicas
    # ------------------------------------------------------------------ #
    def _rule_based(self, df: pd.DataFrame) -> list[Finding]:
        findings: list[Finding] = []
        for col in self._feature_columns:
            low_crit, low, high, high_crit = PHYSIOLOGICAL_RANGES[col]
            for _, row in df.iterrows():
                value = row[col]
                if pd.isna(value):
                    continue
                severity = self._range_severity(value, low_crit, low, high, high_crit)
                if severity is Severity.INFO:
                    continue
                direction = "abaixo" if value < low else "acima"
                findings.append(
                    Finding(
                        modality=Modality.VITALS,
                        severity=severity,
                        description=(
                            f"{_LABELS[col].capitalize()} {direction} do esperado: "
                            f"{value:.1f}"
                        ),
                        score=self._range_score(value, low_crit, low, high, high_crit),
                        timestamp=row["timestamp"],
                        metadata={"signal": col, "value": float(value), "method": "rule"},
                    )
                )
        return findings

    @staticmethod
    def _range_severity(
        value: float, low_crit: float, low: float, high: float, high_crit: float
    ) -> Severity:
        if value <= low_crit or value >= high_crit:
            return Severity.CRITICAL
        if value < low or value > high:
            return Severity.MEDIUM
        return Severity.INFO

    @staticmethod
    def _range_score(
        value: float, low_crit: float, low: float, high: float, high_crit: float
    ) -> float:
        """Distância normalizada da faixa segura, saturada em [0,1]."""
        if value < low:
            span = max(low - low_crit, 1e-6)
            return float(min((low - value) / span, 1.0))
        if value > high:
            span = max(high_crit - high, 1e-6)
            return float(min((value - high) / span, 1.0))
        return 0.0

    # ------------------------------------------------------------------ #
    # 2. Z-score móvel (mudança abrupta vs. linha de base)
    # ------------------------------------------------------------------ #
    def _zscore(self, df: pd.DataFrame) -> list[Finding]:
        findings: list[Finding] = []
        for col in self._feature_columns:
            series = df[col].astype(float)
            roll = series.rolling(self.window, min_periods=max(5, self.window // 2))
            mean = roll.mean().shift(1)
            std = roll.std().shift(1)
            z = (series - mean) / std.replace(0, np.nan)

            for idx, zval in z.items():
                if pd.isna(zval) or abs(zval) < self.zscore_threshold:
                    continue
                # não duplicar quando a regra fisiológica já é crítica
                findings.append(
                    Finding(
                        modality=Modality.VITALS,
                        severity=Severity.MEDIUM if abs(zval) < 4.5 else Severity.HIGH,
                        description=(
                            f"Variação abrupta de {_LABELS[col]} "
                            f"(z={zval:+.1f}) em relação à linha de base"
                        ),
                        score=float(min(abs(zval) / 6.0, 1.0)),
                        timestamp=df.loc[idx, "timestamp"],
                        metadata={
                            "signal": col,
                            "value": float(series[idx]),
                            "zscore": float(zval),
                            "method": "zscore",
                        },
                    )
                )
        return findings

    # ------------------------------------------------------------------ #
    # 3. IsolationForest multivariado
    # ------------------------------------------------------------------ #
    def _isolation_forest(self, df: pd.DataFrame) -> list[Finding]:
        features = df[self._feature_columns].astype(float)
        features = features.interpolate().bfill().ffill()
        if len(features) < 10 or features.isna().all().any():
            return []

        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:  # pragma: no cover - scikit-learn é dependência core
            logger.warning("scikit-learn ausente; pulando IsolationForest.")
            return []

        model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=200,
        )
        labels = model.fit_predict(features.values)
        raw_scores = -model.score_samples(features.values)  # maior = mais anômalo
        norm = self._normalize(raw_scores)

        findings: list[Finding] = []
        for idx, (label, score) in enumerate(zip(labels, norm, strict=True)):
            if label != -1:
                continue
            snapshot = {c: float(features.iloc[idx][c]) for c in self._feature_columns}
            findings.append(
                Finding(
                    modality=Modality.VITALS,
                    severity=Severity.HIGH if score > 0.7 else Severity.MEDIUM,
                    description=(
                        "Combinação atípica de sinais vitais detectada "
                        "(padrão multivariado incomum)"
                    ),
                    score=float(score),
                    timestamp=df.loc[idx, "timestamp"],
                    metadata={"method": "isolation_forest", "snapshot": snapshot},
                )
            )
        return findings

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        lo, hi = float(np.min(x)), float(np.max(x))
        if hi - lo < 1e-9:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)
