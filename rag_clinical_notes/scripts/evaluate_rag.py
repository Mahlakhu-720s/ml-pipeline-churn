"""
Run RAGAS evaluation on the RAG pipeline and log results to MLflow.

Usage:
    python -m rag_clinical_notes.scripts.evaluate_rag
    python -m rag_clinical_notes.scripts.evaluate_rag --eval-set path/to/pairs.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def load_config() -> dict:
    config_path = Path("rag_clinical_notes/configs/rag_config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def main(eval_set_path: str | None = None, run_name: str = "ragas_eval") -> None:
    config = load_config()
    eval_set_path = eval_set_path or config["evaluation"]["test_set_path"]

    vs_cfg = config["vectorstore"]
    emb_cfg = config["embeddings"]
    ret_cfg = config["retrieval"]

    from rag_clinical_notes.src.embeddings.vectorstore import (
        get_embedding_model,
        get_or_create_collection,
    )
    from rag_clinical_notes.src.retrieval.reranker import get_reranker
    from rag_clinical_notes.src.chain.rag_chain import build_rag_chain
    from rag_clinical_notes.src.guardrails.pii_filter import get_analyzer
    from rag_clinical_notes.src.evaluation.ragas_eval import (
        load_eval_set,
        run_evaluation,
        log_to_mlflow,
    )

    print("Loading models and vector store...")
    embedding_model = get_embedding_model(emb_cfg["model"])
    _, collection = get_or_create_collection(
        vs_cfg["persist_directory"],
        vs_cfg["collection_name"],
        vs_cfg["distance_metric"],
    )
    reranker = get_reranker(ret_cfg["reranker_model"])
    chain = build_rag_chain(collection, embedding_model, reranker, config)
    analyzer = get_analyzer()

    print(f"Loading eval set: {eval_set_path}")
    eval_pairs = load_eval_set(eval_set_path)
    print(f"  {len(eval_pairs)} Q&A pairs loaded")

    print("\nRunning evaluation...")
    results = run_evaluation(eval_pairs, chain, analyzer, config)
    metrics = results["metrics"]

    print("\nLogging to MLflow...")
    run_id = log_to_mlflow(metrics, results["per_question"], config, run_name=run_name)

    print("\n" + "=" * 50)
    print("RAGAS Evaluation Results")
    print("=" * 50)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<30} {v:.4f}")
        else:
            print(f"  {k:<30} {v}")
    print(f"\n  MLflow run_id: {run_id}")
    print("  Run `mlflow ui` to view results in the browser.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on the RAG pipeline")
    parser.add_argument("--eval-set", default=None)
    parser.add_argument("--run-name", default="ragas_eval")
    args = parser.parse_args()
    main(args.eval_set, args.run_name)
