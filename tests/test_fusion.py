"""Testes da fusão multimodal e mapeamento de severidade."""

from __future__ import annotations

from multimodal_monitor.integration.fusion import MultimodalFusion
from multimodal_monitor.schemas import Finding, Modality, ModalityResult, Severity


def _result(modality: Modality, severity: Severity, score: float) -> ModalityResult:
    return ModalityResult(
        modality=modality,
        findings=[
            Finding(modality=modality, severity=severity, description="x", score=score)
        ],
    )


def test_no_alert_when_all_modalities_clean() -> None:
    results = [ModalityResult(modality=Modality.VITALS)]
    score, alert = MultimodalFusion().fuse("PAC-1", results)
    assert score == 0.0
    assert alert is None


def test_single_critical_modality_triggers_alert() -> None:
    results = [_result(Modality.VITALS, Severity.CRITICAL, 1.0)]
    score, alert = MultimodalFusion().fuse("PAC-1", results)
    assert alert is not None
    assert alert.severity in (Severity.HIGH, Severity.CRITICAL)


def test_corroboration_raises_score() -> None:
    fusion = MultimodalFusion()
    single = [_result(Modality.VITALS, Severity.MEDIUM, 0.5)]
    multi = [
        _result(Modality.VITALS, Severity.MEDIUM, 0.5),
        _result(Modality.AUDIO, Severity.MEDIUM, 0.5),
        _result(Modality.VIDEO, Severity.MEDIUM, 0.5),
    ]
    score_single, _ = fusion.fuse("PAC-1", single)
    score_multi, _ = fusion.fuse("PAC-1", multi)
    assert score_multi > score_single


def test_score_bounded() -> None:
    results = [
        _result(Modality.VITALS, Severity.CRITICAL, 1.0),
        _result(Modality.AUDIO, Severity.CRITICAL, 1.0),
        _result(Modality.VIDEO, Severity.CRITICAL, 1.0),
        _result(Modality.PRESCRIPTION, Severity.CRITICAL, 1.0),
    ]
    score, _ = MultimodalFusion().fuse("PAC-1", results)
    assert 0.0 <= score <= 1.0
