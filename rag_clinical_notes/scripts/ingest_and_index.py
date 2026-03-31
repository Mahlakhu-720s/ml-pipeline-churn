"""
Ingest clinical PDFs and build the ChromaDB vector index.

Usage:
    python -m rag_clinical_notes.scripts.ingest_and_index
    python -m rag_clinical_notes.scripts.ingest_and_index --strategy semantic
    python -m rag_clinical_notes.scripts.ingest_and_index --reset-collection
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlflow
import yaml


def load_config() -> dict:
    config_path = Path("rag_clinical_notes/configs/rag_config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def main(
    data_dir: str | None = None,
    strategy: str | None = None,
    reset_collection: bool = False,
) -> None:
    config = load_config()

    data_dir = data_dir or config["ingestion"]["data_dir"]
    strategy = strategy or config["chunking"]["strategy"]
    chunking_cfg = config["chunking"].get(strategy, {})

    vs_cfg = config["vectorstore"]
    emb_cfg = config["embeddings"]
    mlflow_cfg = config["mlflow"]

    from rag_clinical_notes.src.ingestion.pdf_loader import load_directory
    from rag_clinical_notes.src.chunking.strategies import chunk_documents
    from rag_clinical_notes.src.embeddings.vectorstore import (
        get_embedding_model,
        get_or_create_collection,
        embed_and_upsert,
    )

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    with mlflow.start_run(run_name="ingestion") as run:
        t0 = time.time()

        print(f"Loading PDFs from: {data_dir}")
        docs = load_directory(Path(data_dir))
        print(f"  Loaded {len(docs)} documents")

        print(f"Chunking with strategy: {strategy}")
        kwargs = {}
        if strategy == "fixed":
            kwargs["chunk_size"] = chunking_cfg.get("chunk_size", 512)
            kwargs["chunk_overlap"] = chunking_cfg.get("chunk_overlap", 50)
        chunks = chunk_documents(docs, strategy=strategy, **kwargs)
        print(f"  Created {len(chunks)} chunks")

        print("Loading embedding model...")
        embedding_model = get_embedding_model(emb_cfg["model"])

        print(f"Setting up ChromaDB at: {vs_cfg['persist_directory']}")
        import chromadb
        client = chromadb.PersistentClient(path=vs_cfg["persist_directory"])

        if reset_collection:
            try:
                client.delete_collection(vs_cfg["collection_name"])
                print(f"  Deleted existing collection: {vs_cfg['collection_name']}")
            except Exception:
                pass

        collection = client.get_or_create_collection(
            name=vs_cfg["collection_name"],
            metadata={"hnsw:space": vs_cfg["distance_metric"]},
        )

        print("Embedding and upserting chunks...")
        count = embed_and_upsert(chunks, collection, embedding_model, emb_cfg["batch_size"])
        elapsed = time.time() - t0

        # Log to MLflow
        mlflow.log_params({
            "strategy": strategy,
            "embeddings_model": emb_cfg["model"],
            "collection_name": vs_cfg["collection_name"],
        })
        mlflow.log_metrics({
            "n_docs": len(docs),
            "n_chunks": len(chunks),
            "upserted": count,
            "elapsed_seconds": round(elapsed, 2),
        })

        print(f"\nIngestion complete in {elapsed:.1f}s")
        print(f"  Documents: {len(docs)}")
        print(f"  Chunks:    {len(chunks)}")
        print(f"  Upserted:  {count}")
        print(f"  MLflow run: {run.info.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest clinical PDFs into ChromaDB")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--strategy", choices=["fixed", "semantic"], default=None)
    parser.add_argument("--reset-collection", action="store_true")
    args = parser.parse_args()
    main(args.data_dir, args.strategy, args.reset_collection)
