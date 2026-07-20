"""Testes da geração de hipóteses interpretativas (apoio à decisão)."""

from __future__ import annotations

from multimodal_monitor.integration.fusion import MultimodalFusion
from multimodal_monitor.integration.hypotheses import generate_hypotheses
from multimodal_monitor.schemas import Finding, Modality, ModalityResult, Severity


def _finding(modality: Modality, severity: Severity = Severity.HIGH, **metadata) -> Finding:
    return Finding(
        modality=modality,
        severity=severity,
        description="x",
        score=0.8,
        metadata=metadata,
    )


def test_no_findings_no_hypotheses() -> None:
    assert generate_hypotheses([]) == []


def test_horizontal_posture_hypothesis() -> None:
    findings = [_finding(Modality.VIDEO, trunk_angle=91.0)]
    hyps = generate_hypotheses(findings)
    assert any("horizontal" in h for h in hyps)
    assert any("queda" in h for h in hyps)


def test_dyspnea_plus_desaturation_combo_comes_first() -> None:
    findings = [
        _finding(Modality.AUDIO, term="falta de ar"),
        _finding(Modality.VITALS, signal="spo2", value=84.0, method="rule"),
    ]
    hyps = generate_hypotheses(findings)
    assert hyps, "esperava hipótese do combo respiratório"
    assert "corroborada" in hyps[0]
    assert "respiratória" in hyps[0]


def test_drug_interaction_hypothesis() -> None:
    findings = [
        _finding(Modality.PRESCRIPTION, rule="interaction", pair=["warfarina", "ibuprofeno"])
    ]
    hyps = generate_hypotheses(findings)
    assert any("Interação medicamentosa" in h for h in hyps)


def test_spike_plus_horizontal_suggests_fall() -> None:
    findings = [
        _finding(Modality.VIDEO, rule="spike", zscore=5.0),
        _finding(Modality.VIDEO, trunk_angle=90.0),
    ]
    hyps = generate_hypotheses(findings)
    assert any("queda não assistida" in h for h in hyps)


def test_hypotheses_capped() -> None:
    # dispara muitos padrões simultâneos e confere o teto
    findings = [
        _finding(Modality.VIDEO, trunk_angle=91.0),
        _finding(Modality.MOVEMENT, rule="immobility"),
        _finding(Modality.VITALS, signal="spo2", value=84.0),
        _finding(Modality.VITALS, signal="heart_rate", value=140.0),
        _finding(Modality.VITALS, signal="systolic_bp", value=190.0),
        _finding(Modality.VITALS, signal="temperature", value=39.0),
        _finding(Modality.AUDIO, term="falta de ar"),
        _finding(Modality.AUDIO, term="tontura"),
        _finding(Modality.PRESCRIPTION, rule="interaction"),
        _finding(Modality.PRESCRIPTION, rule="dose_jump"),
    ]
    hyps = generate_hypotheses(findings)
    assert 0 < len(hyps) <= 5


def test_alert_carries_hypotheses_through_fusion() -> None:
    results = [
        ModalityResult(
            modality=Modality.VITALS,
            findings=[
                _finding(
                    Modality.VITALS,
                    severity=Severity.CRITICAL,
                    signal="spo2",
                    value=82.0,
                    method="rule",
                )
            ],
        ),
        ModalityResult(
            modality=Modality.AUDIO,
            findings=[_finding(Modality.AUDIO, term="falta de ar")],
        ),
    ]
    _score, alert = MultimodalFusion().fuse("PAC-H", results)
    assert alert is not None
    assert alert.hypotheses, "alerta deveria carregar hipóteses"
    assert any("respiratória" in h for h in alert.hypotheses)
    # to_human inclui o bloco de hipóteses com o aviso
    human = alert.to_human()
    assert "não é diagnóstico" in human
