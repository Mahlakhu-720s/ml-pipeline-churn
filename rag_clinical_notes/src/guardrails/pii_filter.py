"""PII detection, redaction, and hallucination checking for RAG guardrails."""
from __future__ import annotations

import functools
import re
from typing import List, Optional, Tuple

REFUSAL_MESSAGE = (
    "I cannot provide an answer grounded in the supplied clinical notes. "
    "The response may not be supported by the provided context. "
    "Please consult a qualified clinician for medical information."
)

_DEFAULT_ENTITIES = [
    "PERSON",
    "DATE_TIME",
    "LOCATION",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "US_SSN",
    "MEDICAL_LICENSE",
]


@functools.lru_cache(maxsize=1)
def get_analyzer():
    """Build and cache a presidio AnalyzerEngine with spacy en_core_web_sm."""
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        })
        nlp_engine = provider.create_engine()
        return AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
    except Exception as e:
        raise RuntimeError(
            f"Failed to load presidio AnalyzerEngine. "
            f"Ensure spacy and en_core_web_sm are installed: {e}"
        )


def detect_pii(
    text: str,
    entities: Optional[List[str]] = None,
    language: str = "en",
) -> list:
    """Run presidio analysis on text. Returns list of RecognizerResult."""
    analyzer = get_analyzer()
    return analyzer.analyze(
        text=text,
        entities=entities or _DEFAULT_ENTITIES,
        language=language,
    )


def redact_pii(
    text: str,
    replacement: str = "[REDACTED]",
    entities: Optional[List[str]] = None,
) -> Tuple[str, list]:
    """Replace all detected PII spans with replacement token.

    Returns (redacted_text, findings_list). findings_list allows logging
    what was redacted without storing the PII itself.
    """
    findings = detect_pii(text, entities=entities)
    if not findings:
        return text, []

    # Sort by start position descending so replacements don't shift offsets
    sorted_findings = sorted(findings, key=lambda r: r.start, reverse=True)
    redacted = text
    for result in sorted_findings:
        redacted = redacted[: result.start] + replacement + redacted[result.end :]

    return redacted, findings


def check_hallucination(
    answer: str,
    context_chunks: List[str],
    min_overlap_ratio: float = 0.3,
) -> Tuple[bool, float]:
    """Lightweight hallucination heuristic based on unigram overlap.

    Computes the ratio of answer words that appear in at least one context chunk.
    Returns (is_grounded: bool, overlap_ratio: float).

    This is a cheap proxy; full faithfulness is computed offline by RAGAS.
    """
    if not answer.strip() or not context_chunks:
        return False, 0.0

    def tokenise(t: str) -> set:
        return set(re.findall(r"\b[a-zA-Z]{3,}\b", t.lower()))

    answer_tokens = tokenise(answer)
    if not answer_tokens:
        return True, 1.0

    context_tokens: set = set()
    for chunk in context_chunks:
        context_tokens |= tokenise(chunk)

    overlap = answer_tokens & context_tokens
    ratio = len(overlap) / len(answer_tokens)
    return ratio >= min_overlap_ratio, ratio


def filter_output(
    answer: str,
    context_chunks: List[str],
    min_overlap_ratio: float = 0.3,
) -> Tuple[str, dict]:
    """Run post-LLM guardrail checks.

    Steps:
    1. PII redaction on the answer.
    2. Hallucination / grounding check.

    Returns (filtered_answer, guardrail_report).
    If not grounded, returns a standardised refusal message instead.
    """
    # Step 1: redact PII from answer
    redacted_answer, pii_findings = redact_pii(answer)

    # Step 2: hallucination check on redacted answer
    is_grounded, overlap_ratio = check_hallucination(
        redacted_answer, context_chunks, min_overlap_ratio
    )

    answer_blocked = not is_grounded
    final_answer = REFUSAL_MESSAGE if answer_blocked else redacted_answer

    report = {
        "pii_found": len(pii_findings) > 0,
        "pii_count": len(pii_findings),
        "is_grounded": is_grounded,
        "overlap_ratio": round(overlap_ratio, 4),
        "answer_blocked": answer_blocked,
    }
    return final_answer, report
