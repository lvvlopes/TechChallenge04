"""Integração multimodal: fusão, alertas e orquestração (RF01+RF02+RF03)."""

from __future__ import annotations

from .alerts import AlertManager
from .fusion import MultimodalFusion
from .orchestrator import PatientMonitor

__all__ = ["MultimodalFusion", "AlertManager", "PatientMonitor"]
