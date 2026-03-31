"""RAGAS evaluation metrics + MLflow logging for the RAG pipeline."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_eval_set(path: str) -> List[dict]:
    """Load JSON evaluation pairs. Returns list of dicts."""
    with open(path) as f:
        return json.load(f)


def run_evaluation(
    eval_pairs: List[dict],
    chain: Any,
    analyzer: Any,
    config: dict,
) -> Dict[str, Any]:
    """Run each eval pair through the chain and compute RAGAS metrics.

    Returns dict of metric name → float score, plus per-question answers.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from langchain_ollama import ChatOllama
    from langchain_huggingface import HuggingFaceEmbeddings

    from rag_clinical_notes.src.chain.rag_chain import run_query

    questions, answers, contexts, ground_truths, per_question = [], [], [], [], []
    total_latency = 0.0
    total_pii = 0

    print(f"  Running {len(eval_pairs)} evaluation queries...")
    for pair in eval_pairs:
        result = run_query(chain, pair["question"], analyzer, config)
        total_latency += result["latency_ms"]
        total_pii += result["guardrail_report"].get("pii_count", 0)

        questions.append(pair["question"])
        answers.append(result["answer"])
        contexts.append([c["text"] for c in result["source_chunks"]])
        ground_truths.append(pair.get("ground_truth", ""))
        per_question.append(
            {
                "question": pair["question"],
                "answer": result["answer"],
                "source_doc_id": pair.get("source_doc_id", ""),
                "latency_ms": result["latency_ms"],
                "guardrail_report": result["guardrail_report"],
            }
        )
        print(f"    [{pair['source_doc_id']}] answered in {result['latency_ms']:.0f}ms")

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    llm_cfg = config.get("llm", {})
    judge_llm = ChatOllama(
        model=llm_cfg.get("model", "llama3"),
        base_url=llm_cfg.get("base_url", "http://localhost:11434"),
        temperature=0.0,
    )
    emb_cfg = config.get("embeddings", {})
    judge_embeddings = HuggingFaceEmbeddings(model_name=emb_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2"))

    print("  Computing RAGAS metrics (this may take a few minutes)...")
    ragas_result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    metrics = {
        "faithfulness": float(ragas_result["faithfulness"]),
        "answer_relevancy": float(ragas_result["answer_relevancy"]),
        "context_precision": float(ragas_result["context_precision"]),
        "context_recall": float(ragas_result["context_recall"]),
        "avg_latency_ms": round(total_latency / max(len(eval_pairs), 1), 2),
        "pii_detections_total": total_pii,
        "n_eval_pairs": len(eval_pairs),
    }

    return {"metrics": metrics, "per_question": per_question}


def log_to_mlflow(
    metrics: dict,
    per_question: List[dict],
    config: dict,
    run_name: str = "ragas_eval",
) -> str:
    """Log RAGAS metrics and artifacts to MLflow.

    Returns the MLflow run_id.
    """
    import mlflow

    mlflow_cfg = config.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlruns/mlflow.db")
    experiment_name = config.get("evaluation", {}).get(
        "mlflow_experiment", "rag_clinical_notes_eval"
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    retrieval_cfg = config.get("retrieval", {})
    llm_cfg = config.get("llm", {})
    chunking_cfg = config.get("chunking", {})

    with mlflow.start_run(run_name=run_name) as run:
        # Parameters
        mlflow.log_params({
            "model": llm_cfg.get("model", "llama3"),
            "chunking_strategy": chunking_cfg.get("strategy", "fixed"),
            "top_k_initial": retrieval_cfg.get("top_k_initial", 20),
            "top_k_final": retrieval_cfg.get("top_k_final", 5),
            "reranker_model": retrieval_cfg.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            "embeddings_model": config.get("embeddings", {}).get("model", ""),
        })

        # Metrics
        mlflow.log_metrics({
            "faithfulness": metrics["faithfulness"],
            "answer_relevancy": metrics["answer_relevancy"],
            "context_precision": metrics["context_precision"],
            "context_recall": metrics["context_recall"],
            "avg_latency_ms": metrics["avg_latency_ms"],
            "pii_detections_total": float(metrics["pii_detections_total"]),
        })

        # Artifacts
        answers_path = "/tmp/ragas_answers.json"
        with open(answers_path, "w") as f:
            json.dump(per_question, f, indent=2)
        mlflow.log_artifact(answers_path, artifact_path="evaluation")

        config_path = "/tmp/rag_config_snapshot.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        mlflow.log_artifact(config_path, artifact_path="evaluation")

        run_id = run.info.run_id

    return run_id
