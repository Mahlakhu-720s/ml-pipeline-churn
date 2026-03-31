"""PDF ingestion: load clinical PDFs into RawDocument objects."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pdfplumber


@dataclass
class RawDocument:
    doc_id: str
    source_path: str
    text: str
    page_count: int
    metadata: dict = field(default_factory=dict)


def load_pdf(pdf_path: Path) -> RawDocument:
    """Extract text from a single PDF using pdfplumber.

    Pages are joined with a form-feed sentinel (\\f) so downstream
    chunkers can split on page boundaries if needed.

    Raises:
        FileNotFoundError: if pdf_path does not exist.
        ValueError: if no text could be extracted.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages_text: List[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            extracted = page.extract_text() or ""
            pages_text.append(extracted.strip())

    full_text = "\f".join(pages_text)
    if not full_text.strip():
        raise ValueError(f"No text extracted from {pdf_path}")

    doc_id = pdf_path.stem
    return RawDocument(
        doc_id=doc_id,
        source_path=str(pdf_path.resolve()),
        text=full_text,
        page_count=page_count,
        metadata={
            "source": str(pdf_path.resolve()),
            "doc_id": doc_id,
            "page_count": page_count,
        },
    )


def load_directory(data_dir: Path) -> List[RawDocument]:
    """Load all PDFs in data_dir, sorted by doc_id for determinism."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    pdf_paths = sorted(data_dir.glob("*.pdf"))
    if not pdf_paths:
        raise ValueError(f"No PDF files found in {data_dir}")

    docs = []
    for path in pdf_paths:
        try:
            doc = load_pdf(path)
            docs.append(doc)
        except (ValueError, Exception) as e:
            print(f"  Warning: skipping {path.name} — {e}")

    return docs
