# Healthcare RAG System — Clinical Notes Q&A

A production-grade Retrieval-Augmented Generation (RAG) system that answers clinical questions from discharge summaries. Built to showcase skills required for the **Netcare AI Engineer** role.

---

## Architecture

```
PDF Discharge Summaries
         │
         ▼
   pdfplumber ingestion ──► RawDocument
         │
         ▼
  Fixed / Semantic Chunking (LangChain)
         │
         ▼
  SentenceTransformer Embeddings ──► ChromaDB (persistent, local)
         │
         ▼
  Query ──► ANN Search (top-20) ──► Cross-Encoder Re-rank (top-5)
                                           │
                          PII Redact (pre-LLM) ──► Ollama Llama3
                                                        │
                                    PII Redact + Hallucination Check (post-LLM)
                                                        │
                                         MLflow Inference Logging
                                                        │
                                              FastAPI Response
                                                        │
                                  RAGAS Eval (faithfulness, relevancy,
                                              context precision/recall)
```

---

## Skills Demonstrated

| Skill | Implementation |
|---|---|
| **RAG** | PDF → chunk → embed → retrieve → re-rank → generate |
| **Chunking strategies** | Fixed-size (`RecursiveCharacterTextSplitter`) + Semantic (`SemanticChunker`) |
| **Vector search** | ChromaDB with cosine similarity + HNSW index |
| **Re-ranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` cross-encoder |
| **LangChain** | Custom `BaseRetriever`, `RetrievalQA`, `ChatOllama` |
| **LLMOps** | MLflow: inference logging, RAGAS metrics, param/metric tracking |
| **Guardrails** | Presidio PII detection/redaction, unigram hallucination check, refusal logic |
| **Responsible AI** | Pre-LLM query sanitisation, post-LLM output filtering |
| **Model serving** | FastAPI with Pydantic v2 schemas, `/health` + `/query` endpoints |
| **Testing** | pytest: chunking, guardrails, API (mocked chain, no real LLM needed) |

---

## Prerequisites

**1. Python 3.10+**

```bash
pip install -r rag_clinical_notes/requirements.txt
python -m spacy download en_core_web_sm
```

**2. Ollama** (free, local LLM — no API key needed)

```bash
# Install: https://ollama.com
ollama pull llama3
ollama serve   # runs on http://localhost:11434
```

---

## Quick Start

### Step 1 — Generate synthetic clinical PDFs

```bash
cd /path/to/ml-pipeline-churn
python -m rag_clinical_notes.scripts.generate_sample_data
```

Creates 5 synthetic discharge summary PDFs in `rag_clinical_notes/data/sample_notes/` and 10 RAGAS evaluation Q&A pairs.

### Step 2 — Ingest and index

```bash
python -m rag_clinical_notes.scripts.ingest_and_index
# Optional: use semantic chunking
python -m rag_clinical_notes.scripts.ingest_and_index --strategy semantic
# Optional: rebuild index from scratch
python -m rag_clinical_notes.scripts.ingest_and_index --reset-collection
```

### Step 3 — Start the API

```bash
uvicorn rag_clinical_notes.src.serving.api:app --port 8001 --reload
```

### Step 4 — Query the system

```bash
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What medications was the diabetic patient discharged with?", "top_k": 5}'
```

Example response:
```json
{
  "query_id": "f3a1c...",
  "question": "What medications was the diabetic patient discharged with?",
  "answer": "Insulin Glargine 20 units nocte, Insulin Aspart sliding scale before meals, Lisinopril 10mg daily, Amlodipine 5mg daily, and Atorvastatin 40mg nocte. Metformin was discontinued.",
  "source_chunks": [...],
  "guardrail_report": {
    "pii_found": false,
    "pii_count": 0,
    "is_grounded": true,
    "overlap_ratio": 0.78,
    "answer_blocked": false
  },
  "latency_ms": 1823.4,
  "mlflow_run_id": "abc123..."
}
```

### Step 5 — Run RAGAS evaluation

```bash
python -m rag_clinical_notes.scripts.evaluate_rag
```

Outputs a metrics table and logs results to MLflow:

```
==================================================
RAGAS Evaluation Results
==================================================
  faithfulness                   0.8714
  answer_relevancy               0.9102
  context_precision              0.8433
  context_recall                 0.7891
  avg_latency_ms                 1640.00
  pii_detections_total           0
  n_eval_pairs                   10
```

### Step 6 — View MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
# Open http://localhost:5000
```

Experiments visible:
- `rag_clinical_notes` — ingestion runs + inference logs
- `rag_clinical_notes_eval` — RAGAS evaluation runs

---

## Running Tests

```bash
pytest rag_clinical_notes/tests/ -v
```

Tests do **not** require Ollama or a running server — all LLM calls are mocked.

---

## Project Structure

```
rag_clinical_notes/
├── configs/rag_config.yaml          # All configuration
├── src/
│   ├── ingestion/pdf_loader.py      # PDF → RawDocument
│   ├── chunking/strategies.py       # Fixed + semantic chunking
│   ├── embeddings/vectorstore.py    # ChromaDB + embeddings
│   ├── retrieval/reranker.py        # Cross-encoder re-ranking
│   ├── guardrails/pii_filter.py     # PII detection + hallucination check
│   ├── chain/rag_chain.py           # LangChain chain (Ollama)
│   ├── evaluation/ragas_eval.py     # RAGAS metrics + MLflow
│   └── serving/api.py               # FastAPI endpoint
├── scripts/
│   ├── generate_sample_data.py      # Create synthetic PDFs
│   ├── ingest_and_index.py          # Index PDFs into ChromaDB
│   └── evaluate_rag.py              # Run RAGAS evaluation
├── tests/                           # pytest test suite
├── data/sample_notes/               # Synthetic discharge summaries
└── requirements.txt
```

---

## Sample Patient Profiles (Synthetic)

| Doc ID | Scenario | Key Clinical Content |
|---|---|---|
| `patient_001_diabetes_ckd` | Type 2 DM + CKD Stage 3 | Insulin initiation, Metformin discontinuation, eGFR 38 |
| `patient_002_acute_mi` | Inferior STEMI | Primary PCI to RCA, dual antiplatelet therapy |
| `patient_003_chf_exacerbation` | Decompensated Heart Failure | EF 28%, IV furosemide, Sacubitril/Valsartan |
| `patient_004_pneumonia` | Community-Acquired Pneumonia | Strep pneumoniae, antibiotic de-escalation |
| `patient_005_hip_fracture` | Hip Fracture + Osteoporosis | THR, VTE prophylaxis, bone health management |

---

## Configuration

All settings live in `configs/rag_config.yaml`. Key toggles:

| Setting | Default | Options |
|---|---|---|
| `chunking.strategy` | `fixed` | `fixed`, `semantic` |
| `chunking.fixed.chunk_size` | `512` | any int |
| `retrieval.top_k_initial` | `20` | 5–100 |
| `retrieval.top_k_final` | `5` | 1–20 |
| `llm.model` | `llama3` | any Ollama model |
| `guardrails.min_overlap_ratio` | `0.3` | 0.0–1.0 |
| `serving.port` | `8001` | any free port |
