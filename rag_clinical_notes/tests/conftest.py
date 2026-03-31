"""Pytest fixtures for RAG Clinical Notes tests."""
from __future__ import annotations

from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest

from rag_clinical_notes.src.ingestion.pdf_loader import RawDocument
from rag_clinical_notes.src.chunking.strategies import Chunk

SAMPLE_TEXT = """\
DISCHARGE SUMMARY

Patient: [SYNTHETIC] Test Patient  |  MRN: SYN-99999  |  DOB: 1960-01-01
Admission Date: 2024-01-01  |  Discharge Date: 2024-01-05
Attending Physician: [SYNTHETIC] Dr. Test

CHIEF COMPLAINT:
  Chest pain and shortness of breath.

HISTORY OF PRESENT ILLNESS:
  A 64-year-old male presented with acute onset chest pain radiating to the left arm.
  Associated diaphoresis and nausea were present. No fever or cough.

PAST MEDICAL HISTORY:
  Hypertension, Hypercholesterolaemia, Type 2 Diabetes Mellitus.

MEDICATIONS ON ADMISSION:
  Metformin 1000mg BD, Amlodipine 5mg daily, Atorvastatin 40mg nocte.

PHYSICAL EXAMINATION:
  BP 150/92 mmHg, HR 98 bpm, Temp 36.7 degrees C, SpO2 95% on room air.
  Mild bibasal crackles. No peripheral oedema.

DIAGNOSTIC RESULTS:
  Troponin I 18.4 ng/mL. ECG: ST-elevation in V1-V4.
  CXR: mild cardiomegaly, no pulmonary oedema.
  HbA1c 8.1%, Creatinine 95 umol/L, eGFR 62.

HOSPITAL COURSE:
  Primary PCI performed with drug-eluting stent to LAD.
  Dual antiplatelet therapy started: Aspirin 100mg daily and Ticagrelor 90mg BD.
  Ramipril 2.5mg daily and Bisoprolol 2.5mg daily commenced.
  Glucose management optimised; Metformin continued.

DISCHARGE DIAGNOSIS:
  1. Anterior STEMI, post-PCI with DES to LAD.
  2. Hypertension.
  3. Type 2 Diabetes Mellitus, controlled.

DISCHARGE MEDICATIONS:
  Aspirin 100mg daily, Ticagrelor 90mg BD, Ramipril 2.5mg daily,
  Bisoprolol 2.5mg daily, Atorvastatin 80mg nocte, Amlodipine 5mg daily,
  Metformin 1000mg BD.

FOLLOW-UP INSTRUCTIONS:
  Cardiology OPD in 2 weeks. Cardiac rehabilitation referral placed.
  Do NOT stop antiplatelet therapy without cardiology advice.

CONDITION AT DISCHARGE:
  Stable and haemodynamically compensated.
"""


@pytest.fixture(scope="session")
def rag_config() -> dict:
    """Load rag_config.yaml once for all RAG tests."""
    import yaml
    config_path = Path("rag_clinical_notes/configs/rag_config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def sample_raw_doc() -> RawDocument:
    """A single in-memory RawDocument with synthetic discharge note text."""
    return RawDocument(
        doc_id="test_patient_stemi",
        source_path="/tmp/test_patient_stemi.pdf",
        text=SAMPLE_TEXT,
        page_count=1,
        metadata={
            "source": "/tmp/test_patient_stemi.pdf",
            "doc_id": "test_patient_stemi",
            "page_count": 1,
        },
    )


@pytest.fixture(scope="session")
def fixed_chunks(sample_raw_doc) -> List[Chunk]:
    """Pre-chunked list from fixed_chunk(sample_raw_doc)."""
    from rag_clinical_notes.src.chunking.strategies import fixed_chunk
    return fixed_chunk(sample_raw_doc, chunk_size=256, chunk_overlap=30)


@pytest.fixture
def mock_collection():
    """MagicMock chromadb Collection for API tests (avoids disk I/O)."""
    mock = MagicMock()
    mock.count.return_value = 10
    mock.query.return_value = {
        "ids": [["chunk_001", "chunk_002"]],
        "documents": [["Aspirin 100mg daily was prescribed.", "Ticagrelor 90mg BD was started."]],
        "metadatas": [[
            {"doc_id": "test_doc", "section": "DISCHARGE MEDICATIONS", "chunk_index": 0,
             "chunk_strategy": "fixed", "page_count": 1, "char_start": 0, "char_end": 50,
             "source_path": "/tmp/test.pdf"},
            {"doc_id": "test_doc", "section": "HOSPITAL COURSE", "chunk_index": 1,
             "chunk_strategy": "fixed", "page_count": 1, "char_start": 50, "char_end": 100,
             "source_path": "/tmp/test.pdf"},
        ]],
        "distances": [[0.12, 0.18]],
    }
    return mock


@pytest.fixture
def mock_chain():
    """MagicMock RetrievalQA that returns a deterministic answer."""
    from langchain.schema import Document
    mock = MagicMock()
    mock.invoke.return_value = {
        "result": "Aspirin 100mg daily and Ticagrelor 90mg twice daily were prescribed.",
        "source_documents": [
            Document(
                page_content="Aspirin 100mg daily and Ticagrelor 90mg BD were started.",
                metadata={"doc_id": "test_doc", "section": "DISCHARGE MEDICATIONS",
                          "rerank_score": 0.95, "chunk_strategy": "fixed",
                          "page_count": 1, "char_start": 0, "char_end": 60,
                          "source_path": "/tmp/test.pdf"},
            )
        ],
    }
    return mock
