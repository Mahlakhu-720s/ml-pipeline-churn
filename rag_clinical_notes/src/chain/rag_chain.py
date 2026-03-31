"""LangChain RAG chain with Ollama LLM and re-ranking retriever."""
from __future__ import annotations

import time
import uuid
from typing import Any, List, Optional

from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.schema import BaseRetriever, Document

from rag_clinical_notes.src.embeddings.vectorstore import query_collection
from rag_clinical_notes.src.retrieval.reranker import rerank

SYSTEM_PROMPT = """You are a clinical decision-support assistant.
Your sole function is to answer questions about the clinical discharge notes provided in the context below.

Rules you must never break:
1. Answer ONLY from the provided context. Do not use any prior medical knowledge.
2. If the context does not contain enough information to answer, say exactly:
   "The provided clinical notes do not contain sufficient information to answer this question."
3. Never speculate about diagnoses, prognoses, or treatments not explicitly stated.
4. Never reproduce patient names, dates of birth, MRNs, or any other identifying information.
5. Present information as factual summaries, not medical advice.
6. If asked for advice rather than information, redirect: "Please consult a qualified clinician."

Context (clinical notes):
{context}"""

HUMAN_TEMPLATE = "Question: {question}"


class RerankingRetriever(BaseRetriever):
    """LangChain retriever that wraps ChromaDB ANN search + cross-encoder re-ranking."""

    collection: Any
    embedding_model: Any
    reranker: Any
    top_k_initial: int = 20
    top_k_final: int = 5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        candidates = query_collection(
            query, self.collection, self.embedding_model, top_k=self.top_k_initial
        )
        reranked = rerank(query, candidates, self.reranker, top_k=self.top_k_final)
        return [
            Document(page_content=r["text"], metadata=r["metadata"] | {"rerank_score": r["rerank_score"]})
            for r in reranked
        ]


def build_rag_chain(
    collection: Any,
    embedding_model: Any,
    reranker: Any,
    config: dict,
) -> RetrievalQA:
    """Assemble the full LangChain RetrievalQA chain backed by Ollama."""
    from langchain_ollama import ChatOllama

    llm_cfg = config.get("llm", {})
    retrieval_cfg = config.get("retrieval", {})

    llm = ChatOllama(
        model=llm_cfg.get("model", "llama3"),
        base_url=llm_cfg.get("base_url", "http://localhost:11434"),
        temperature=llm_cfg.get("temperature", 0.0),
        num_predict=llm_cfg.get("max_tokens", 1024),
    )

    retriever = RerankingRetriever(
        collection=collection,
        embedding_model=embedding_model,
        reranker=reranker,
        top_k_initial=retrieval_cfg.get("top_k_initial", 20),
        top_k_final=retrieval_cfg.get("top_k_final", 5),
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE),
    ])

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return chain


def run_query(
    chain: RetrievalQA,
    raw_query: str,
    analyzer: Any,
    config: dict,
) -> dict:
    """Run the full RAG pipeline for a single query.

    Steps:
    1. Redact PII from the raw query (pre-LLM).
    2. Invoke the chain.
    3. Apply post-LLM guardrails (PII redaction + hallucination check).

    Returns a result dict with query metadata, answer, and guardrail report.
    """
    from rag_clinical_notes.src.guardrails.pii_filter import filter_output, redact_pii

    guardrail_cfg = config.get("guardrails", {})
    min_overlap = guardrail_cfg.get("min_overlap_ratio", 0.3)

    # Pre-LLM: sanitise the query
    sanitised_query, _ = redact_pii(raw_query)

    t0 = time.time()
    result = chain.invoke({"query": sanitised_query})
    latency_ms = (time.time() - t0) * 1000

    raw_answer = result.get("result", "")
    source_docs = result.get("source_documents", [])
    context_texts = [doc.page_content for doc in source_docs]

    # Post-LLM: guardrail filtering
    final_answer, guardrail_report = filter_output(
        raw_answer, context_texts, min_overlap_ratio=min_overlap
    )

    source_chunks = [
        {
            "chunk_id": doc.metadata.get("chunk_id", doc.metadata.get("doc_id", "")),
            "doc_id": doc.metadata.get("doc_id", ""),
            "section": doc.metadata.get("section", "UNKNOWN"),
            "text": doc.page_content,
            "rerank_score": doc.metadata.get("rerank_score", 0.0),
        }
        for doc in source_docs
    ]

    llm_cfg = config.get("llm", {})
    return {
        "query_id": str(uuid.uuid4()),
        "original_query": raw_query,
        "sanitised_query": sanitised_query,
        "answer": final_answer,
        "source_chunks": source_chunks,
        "guardrail_report": guardrail_report,
        "latency_ms": round(latency_ms, 2),
        "model": llm_cfg.get("model", "llama3"),
    }
