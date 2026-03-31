"""Tests for the FastAPI RAG serving endpoint."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from rag_clinical_notes.src.serving.api import app, app_state


@pytest.fixture(autouse=True)
def clear_app_state():
    """Reset app state between tests."""
    app_state.clear()
    yield
    app_state.clear()


@pytest.fixture
def loaded_app_state(mock_collection, mock_chain):
    """Populate app_state as if lifespan startup completed."""
    from unittest.mock import MagicMock
    app_state["config"] = {
        "llm": {"model": "llama3", "base_url": "http://localhost:11434"},
        "retrieval": {"top_k_initial": 20, "top_k_final": 5},
        "guardrails": {"min_overlap_ratio": 0.3},
        "mlflow": {"tracking_uri": "sqlite:///mlruns/test.db", "experiment_name": "test"},
        "embeddings": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
        "vectorstore": {"persist_directory": "/tmp/test_chroma", "collection_name": "test",
                        "distance_metric": "cosine"},
    }
    app_state["collection"] = mock_collection
    app_state["chain"] = mock_chain
    app_state["analyzer"] = MagicMock()
    app_state["embedding_model"] = MagicMock()
    app_state["reranker"] = MagicMock()


@pytest.mark.asyncio
async def test_health_returns_ok(loaded_app_state):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["chain_loaded"] is True
    assert data["collection_loaded"] is True
    assert data["document_count"] == 10


@pytest.mark.asyncio
async def test_health_degraded_when_chain_not_loaded():
    # app_state is empty (cleared by autouse fixture)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["chain_loaded"] is False


@pytest.mark.asyncio
async def test_query_503_when_chain_not_loaded():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/query", json={"question": "What medications were prescribed?"}
        )
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_query_validates_min_length(loaded_app_state):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/query", json={"question": "Hi"})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_query_validates_top_k_range(loaded_app_state):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/query", json={"question": "What medications were given?", "top_k": 0}
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_query_returns_response_schema(loaded_app_state):
    """Mock run_query so we don't need a real Ollama server."""
    mock_result = {
        "query_id": "abc-123",
        "original_query": "What medications were prescribed?",
        "sanitised_query": "What medications were prescribed?",
        "answer": "Aspirin 100mg daily and Ticagrelor 90mg BD were prescribed.",
        "source_chunks": [
            {
                "chunk_id": "test_doc_fixed_0000",
                "doc_id": "test_doc",
                "section": "DISCHARGE MEDICATIONS",
                "text": "Aspirin 100mg daily and Ticagrelor 90mg BD.",
                "rerank_score": 0.95,
            }
        ],
        "guardrail_report": {
            "pii_found": False,
            "pii_count": 0,
            "is_grounded": True,
            "overlap_ratio": 0.72,
            "answer_blocked": False,
        },
        "latency_ms": 250.0,
        "model": "llama3",
    }

    with patch("rag_clinical_notes.src.serving.api.run_query", return_value=mock_result), \
         patch("mlflow.start_run"), patch("mlflow.set_tracking_uri"), \
         patch("mlflow.set_experiment"), patch("mlflow.log_params"), \
         patch("mlflow.log_metrics"):

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/query", json={"question": "What medications were prescribed?"}
            )

    assert response.status_code == 200
    data = response.json()
    assert "query_id" in data
    assert "answer" in data
    assert "source_chunks" in data
    assert "guardrail_report" in data
    assert "latency_ms" in data
    assert isinstance(data["source_chunks"], list)
