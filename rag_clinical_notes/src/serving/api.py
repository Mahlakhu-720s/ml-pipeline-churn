"""FastAPI endpoint for the Healthcare RAG Clinical Notes Q&A system."""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="Clinical question to ask against the discharge notes.",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context chunks to retrieve.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "What medications was the diabetic patient discharged with?",
                "top_k": 5,
            }
        }
    }


class SourceChunk(BaseModel):
    chunk_id: str
    doc_id: str
    section: str
    text: str
    rerank_score: float


class QueryResponse(BaseModel):
    query_id: str
    question: str
    answer: str
    source_chunks: List[SourceChunk]
    guardrail_report: Dict[str, Any]
    latency_ms: float
    mlflow_run_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    collection_loaded: bool
    chain_loaded: bool
    document_count: int
    uptime_seconds: Optional[float] = None


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

app_state: Dict[str, Any] = {}
_startup_time: float = 0.0


def _load_config() -> dict:
    config_path = Path("rag_clinical_notes/configs/rag_config.yaml")
    if not config_path.exists():
        config_path = Path(__file__).parents[3] / "configs" / "rag_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _startup_time
    _startup_time = time.time()

    config = _load_config()
    app_state["config"] = config

    from rag_clinical_notes.src.embeddings.vectorstore import (
        get_embedding_model,
        get_or_create_collection,
    )
    from rag_clinical_notes.src.retrieval.reranker import get_reranker
    from rag_clinical_notes.src.chain.rag_chain import build_rag_chain
    from rag_clinical_notes.src.guardrails.pii_filter import get_analyzer

    vs_cfg = config["vectorstore"]
    emb_cfg = config["embeddings"]
    ret_cfg = config["retrieval"]

    embedding_model = get_embedding_model(emb_cfg["model"])
    _, collection = get_or_create_collection(
        vs_cfg["persist_directory"],
        vs_cfg["collection_name"],
        vs_cfg["distance_metric"],
    )
    reranker = get_reranker(ret_cfg["reranker_model"])
    chain = build_rag_chain(collection, embedding_model, reranker, config)
    analyzer = get_analyzer()

    app_state.update(
        {
            "collection": collection,
            "embedding_model": embedding_model,
            "reranker": reranker,
            "chain": chain,
            "analyzer": analyzer,
        }
    )

    yield

    app_state.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Healthcare RAG — Clinical Notes Q&A",
    description=(
        "Answers clinical questions from discharge summaries using "
        "RAG + re-ranking + PII guardrails. Portfolio project for Netcare AI Engineer role."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    collection = app_state.get("collection")
    chain = app_state.get("chain")
    doc_count = 0
    if collection is not None:
        try:
            doc_count = collection.count()
        except Exception:
            pass

    uptime = time.time() - _startup_time if _startup_time else None
    return HealthResponse(
        status="ok" if chain is not None else "degraded",
        collection_loaded=collection is not None,
        chain_loaded=chain is not None,
        document_count=doc_count,
        uptime_seconds=round(uptime, 1) if uptime else None,
    )


@app.post("/query", response_model=QueryResponse, tags=["rag"])
async def query_notes(request: QueryRequest):
    chain = app_state.get("chain")
    if chain is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG chain not loaded. Check server logs.",
        )

    from rag_clinical_notes.src.chain.rag_chain import run_query

    config = app_state["config"]

    # Override top_k from request
    config_copy = dict(config)
    config_copy["retrieval"] = dict(config.get("retrieval", {}))
    config_copy["retrieval"]["top_k_final"] = request.top_k

    result = run_query(chain, request.question, app_state["analyzer"], config_copy)

    # Log to MLflow asynchronously (best-effort)
    mlflow_run_id = None
    try:
        import mlflow
        mlflow_cfg = config.get("mlflow", {})
        mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "sqlite:///mlruns/mlflow.db"))
        mlflow.set_experiment(mlflow_cfg.get("experiment_name", "rag_clinical_notes"))
        with mlflow.start_run(run_name="inference") as run:
            mlflow.log_params({"model": result["model"], "top_k": request.top_k})
            mlflow.log_metrics({
                "latency_ms": result["latency_ms"],
                "pii_count": float(result["guardrail_report"].get("pii_count", 0)),
                "overlap_ratio": result["guardrail_report"].get("overlap_ratio", 0.0),
                "answer_blocked": float(result["guardrail_report"].get("answer_blocked", False)),
            })
            mlflow_run_id = run.info.run_id
    except Exception:
        pass  # MLflow logging is non-critical for serving

    return QueryResponse(
        query_id=result["query_id"],
        question=result["original_query"],
        answer=result["answer"],
        source_chunks=[SourceChunk(**c) for c in result["source_chunks"]],
        guardrail_report=result["guardrail_report"],
        latency_ms=result["latency_ms"],
        mlflow_run_id=mlflow_run_id,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    config = _load_config()
    serving_cfg = config.get("serving", {})
    uvicorn.run(
        "rag_clinical_notes.src.serving.api:app",
        host=serving_cfg.get("host", "0.0.0.0"),
        port=serving_cfg.get("port", 8001),
        reload=False,
    )
