"""Configuração central da aplicação.

Carrega variáveis de ambiente (via `.env`) usando `pydantic-settings`.
Todas as credenciais sensíveis vivem aqui e NUNCA devem ser commitadas.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configurações da aplicação, lidas do ambiente / arquivo `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Ambiente ---
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"
    default_locale: str = "pt-BR"

    # --- Azure Speech to Text (RF02) ---
    azure_speech_key: str = ""
    azure_speech_region: str = "brazilsouth"

    # --- Azure Language / Text Analytics (RF02) ---
    azure_language_key: str = ""
    azure_language_endpoint: str = ""

    # --- Detecção de anomalias (RF03) ---
    anomaly_contamination: float = Field(default=0.02, ge=0.0, le=0.5)
    vitals_zscore_threshold: float = Field(default=3.0, gt=0.0)

    # --- Alertas ---
    alert_channel: Literal["console", "file", "webhook"] = "console"
    alert_webhook_url: str = ""

    # --- Propriedades derivadas (feature flags de degradação graciosa) ---
    @property
    def azure_speech_enabled(self) -> bool:
        """True quando há credenciais válidas para o Azure Speech."""
        return bool(self.azure_speech_key and self.azure_speech_region)

    @property
    def azure_language_enabled(self) -> bool:
        """True quando há credenciais válidas para o Azure Language."""
        return bool(self.azure_language_key and self.azure_language_endpoint)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retorna a instância singleton de configurações.

    Usa cache para evitar reler o ambiente a cada chamada.
    """
    return Settings()
