"""Chunking strategies: fixed-size and semantic chunking."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter

from rag_clinical_notes.src.ingestion.pdf_loader import RawDocument

# ALL-CAPS section headers found in discharge summaries
_SECTION_PATTERN = re.compile(r"^([A-Z][A-Z\s/\-]{3,}):?\s*$", re.MULTILINE)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def infer_section(full_doc_text: str, char_start: int) -> str:
    """Return the most recent ALL-CAPS section header at or before char_start."""
    last_section = "UNKNOWN"
    for match in _SECTION_PATTERN.finditer(full_doc_text):
        if match.start() <= char_start:
            last_section = match.group(1).strip()
        else:
            break
    return last_section


def fixed_chunk(
    doc: RawDocument,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> List[Chunk]:
    """Split doc.text into overlapping fixed-size windows (chars).

    Uses LangChain RecursiveCharacterTextSplitter. chunk_size is in
    characters (chunk_size * 4 approximates tokens for English text).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\f", "\n\n", "\n", ". ", " ", ""],
    )

    raw_chunks = splitter.split_text(doc.text)
    chunks: List[Chunk] = []
    search_start = 0

    for idx, chunk_text in enumerate(raw_chunks):
        char_start = doc.text.find(chunk_text, search_start)
        if char_start == -1:
            char_start = 0
        char_end = char_start + len(chunk_text)
        search_start = max(0, char_end - chunk_overlap)

        section = infer_section(doc.text, char_start)
        chunk_id = f"{doc.doc_id}_fixed_{idx:04d}"

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                text=chunk_text,
                metadata={
                    "doc_id": doc.doc_id,
                    "source_path": doc.source_path,
                    "chunk_index": idx,
                    "chunk_strategy": "fixed",
                    "section": section,
                    "page_count": doc.page_count,
                    "char_start": char_start,
                    "char_end": char_end,
                },
            )
        )

    return chunks


def semantic_chunk(
    doc: RawDocument,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    breakpoint_percentile: int = 95,
) -> List[Chunk]:
    """Split doc.text by semantic similarity using LangChain SemanticChunker.

    Groups consecutive sentences until cosine distance exceeds the percentile
    breakpoint threshold.
    """
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    chunker = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=breakpoint_percentile,
    )

    raw_chunks = chunker.split_text(doc.text)
    chunks: List[Chunk] = []
    search_start = 0

    for idx, chunk_text in enumerate(raw_chunks):
        char_start = doc.text.find(chunk_text, search_start)
        if char_start == -1:
            char_start = 0
        char_end = char_start + len(chunk_text)
        search_start = max(0, char_end - 20)

        section = infer_section(doc.text, char_start)
        chunk_id = f"{doc.doc_id}_semantic_{idx:04d}"

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                text=chunk_text,
                metadata={
                    "doc_id": doc.doc_id,
                    "source_path": doc.source_path,
                    "chunk_index": idx,
                    "chunk_strategy": "semantic",
                    "section": section,
                    "page_count": doc.page_count,
                    "char_start": char_start,
                    "char_end": char_end,
                },
            )
        )

    return chunks


def chunk_documents(
    docs: List[RawDocument],
    strategy: str = "fixed",
    **kwargs,
) -> List[Chunk]:
    """Dispatch chunking across all docs using the given strategy."""
    all_chunks: List[Chunk] = []
    for doc in docs:
        if strategy == "fixed":
            chunks = fixed_chunk(doc, **kwargs)
        elif strategy == "semantic":
            chunks = semantic_chunk(doc, **kwargs)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy!r}. Use 'fixed' or 'semantic'.")
        all_chunks.extend(chunks)
    return all_chunks
