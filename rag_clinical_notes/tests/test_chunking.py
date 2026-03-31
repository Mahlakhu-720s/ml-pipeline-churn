"""Unit tests for chunking strategies."""
from __future__ import annotations

from rag_clinical_notes.src.chunking.strategies import (
    Chunk,
    fixed_chunk,
    infer_section,
)
from rag_clinical_notes.src.ingestion.pdf_loader import RawDocument


def test_fixed_chunk_returns_chunks(sample_raw_doc):
    chunks = fixed_chunk(sample_raw_doc, chunk_size=256, chunk_overlap=30)
    assert len(chunks) >= 1


def test_fixed_chunk_count_reasonable(sample_raw_doc):
    """A ~2000 char document with chunk_size=256 should yield multiple chunks."""
    chunks = fixed_chunk(sample_raw_doc, chunk_size=256, chunk_overlap=30)
    assert len(chunks) >= 3


def test_fixed_chunk_ids_unique(fixed_chunks):
    ids = [c.chunk_id for c in fixed_chunks]
    assert len(ids) == len(set(ids)), "Chunk IDs must be unique"


def test_chunk_metadata_schema(fixed_chunks):
    required_keys = {
        "doc_id", "source_path", "chunk_index", "chunk_strategy",
        "section", "page_count", "char_start", "char_end",
    }
    for chunk in fixed_chunks:
        assert required_keys.issubset(chunk.metadata.keys()), (
            f"Chunk {chunk.chunk_id} missing metadata keys: "
            f"{required_keys - chunk.metadata.keys()}"
        )


def test_chunk_strategy_label(fixed_chunks):
    for chunk in fixed_chunks:
        assert chunk.metadata["chunk_strategy"] == "fixed"


def test_chunk_doc_id_matches(sample_raw_doc, fixed_chunks):
    for chunk in fixed_chunks:
        assert chunk.doc_id == sample_raw_doc.doc_id


def test_chunk_text_non_empty(fixed_chunks):
    for chunk in fixed_chunks:
        assert chunk.text.strip(), f"Chunk {chunk.chunk_id} has empty text"


def test_section_inference_known_section():
    text = "DISCHARGE MEDICATIONS:\n  Aspirin 100mg daily, Ticagrelor 90mg BD."
    section = infer_section(text, char_start=25)
    assert section == "DISCHARGE MEDICATIONS"


def test_section_inference_unknown():
    text = "Some text without any section header."
    section = infer_section(text, char_start=5)
    assert section == "UNKNOWN"


def test_section_inference_picks_most_recent():
    text = (
        "CHIEF COMPLAINT:\n  Chest pain.\n\n"
        "HOSPITAL COURSE:\n  PCI performed.\n\n"
        "DISCHARGE MEDICATIONS:\n  Aspirin 100mg."
    )
    # Should pick HOSPITAL COURSE for a position in the middle
    mid = text.index("PCI performed")
    section = infer_section(text, char_start=mid)
    assert section == "HOSPITAL COURSE"


def test_fixed_chunk_char_positions_valid(sample_raw_doc, fixed_chunks):
    for chunk in fixed_chunks:
        start = chunk.metadata["char_start"]
        end = chunk.metadata["char_end"]
        assert start >= 0
        assert end > start
        assert end <= len(sample_raw_doc.text) + 10  # small tolerance
