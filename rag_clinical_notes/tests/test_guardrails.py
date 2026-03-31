"""Unit tests for PII guardrails and hallucination detection."""
from __future__ import annotations

from rag_clinical_notes.src.guardrails.pii_filter import (
    REFUSAL_MESSAGE,
    check_hallucination,
    filter_output,
    redact_pii,
)


# ---------------------------------------------------------------------------
# PII detection and redaction
# ---------------------------------------------------------------------------

def test_pii_redacted_from_answer():
    """A name in an answer should be redacted."""
    text = "The patient John Smith was discharged on Aspirin."
    redacted, findings = redact_pii(text)
    assert "John Smith" not in redacted
    assert "[REDACTED]" in redacted


def test_no_pii_returns_original():
    """Text without PII should pass through unchanged."""
    text = "Aspirin 100mg daily and Ticagrelor 90mg twice daily were prescribed."
    redacted, findings = redact_pii(text)
    assert redacted == text
    assert findings == []


def test_medical_diagnosis_not_flagged_as_pii():
    """Medical terminology should not be treated as PII."""
    text = "The patient was diagnosed with Type 2 Diabetes Mellitus and Hypertension."
    _, findings = redact_pii(text)
    # No PERSON entity should be triggered by diagnoses
    person_findings = [f for f in findings if f.entity_type == "PERSON"]
    assert len(person_findings) == 0


def test_redact_pii_output_tuple():
    text = "Patient: Jane Doe, DOB: 1980-05-12"
    result = redact_pii(text)
    assert isinstance(result, tuple)
    assert len(result) == 2
    redacted, findings = result
    assert isinstance(redacted, str)
    assert isinstance(findings, list)


# ---------------------------------------------------------------------------
# Hallucination detection
# ---------------------------------------------------------------------------

def test_hallucination_check_grounded():
    """Answer with terms from context should be flagged as grounded."""
    answer = "The patient was prescribed Aspirin and Ticagrelor for antiplatelet therapy."
    context = [
        "Aspirin 100mg daily and Ticagrelor 90mg BD were started after the PCI procedure.",
        "Antiplatelet therapy was initiated in the catheterisation laboratory.",
    ]
    is_grounded, ratio = check_hallucination(answer, context)
    assert is_grounded is True
    assert ratio >= 0.3


def test_hallucination_check_ungrounded():
    """Answer with fabricated medical content not in context should be ungrounded."""
    answer = "The patient underwent kidney transplantation and received cyclosporine immunosuppression."
    context = [
        "Aspirin 100mg daily was prescribed after the PCI.",
        "Bisoprolol 2.5mg daily commenced for rate control.",
    ]
    is_grounded, ratio = check_hallucination(answer, context, min_overlap_ratio=0.6)
    assert is_grounded is False


def test_hallucination_check_empty_answer():
    is_grounded, ratio = check_hallucination("", ["some context text"])
    assert is_grounded is False
    assert ratio == 0.0


def test_hallucination_check_empty_context():
    is_grounded, ratio = check_hallucination("some answer text", [])
    assert is_grounded is False


def test_hallucination_returns_float_ratio():
    answer = "Aspirin and Ticagrelor were prescribed."
    context = ["Aspirin 100mg and Ticagrelor 90mg BD were given."]
    _, ratio = check_hallucination(answer, context)
    assert isinstance(ratio, float)
    assert 0.0 <= ratio <= 1.0


# ---------------------------------------------------------------------------
# filter_output integration
# ---------------------------------------------------------------------------

def test_filter_output_passes_grounded_answer():
    answer = "The patient was discharged on Aspirin 100mg daily and Ticagrelor 90mg BD."
    context = [
        "Aspirin 100mg daily and Ticagrelor 90mg twice daily were prescribed after PCI.",
        "Dual antiplatelet therapy was initiated in the cath lab.",
    ]
    final, report = filter_output(answer, context, min_overlap_ratio=0.3)
    assert final != REFUSAL_MESSAGE
    assert report["answer_blocked"] is False
    assert "is_grounded" in report
    assert "pii_found" in report
    assert "overlap_ratio" in report


def test_filter_output_blocks_hallucinated_answer():
    answer = "The patient received a liver transplant and is on tacrolimus and mycophenolate."
    context = ["Aspirin was prescribed for antiplatelet therapy after PCI."]
    final, report = filter_output(answer, context, min_overlap_ratio=0.9)
    assert final == REFUSAL_MESSAGE
    assert report["answer_blocked"] is True


def test_filter_output_report_keys():
    answer = "Aspirin was prescribed."
    context = ["Aspirin 100mg daily was given."]
    _, report = filter_output(answer, context)
    assert set(report.keys()) == {
        "pii_found", "pii_count", "is_grounded", "overlap_ratio", "answer_blocked"
    }
