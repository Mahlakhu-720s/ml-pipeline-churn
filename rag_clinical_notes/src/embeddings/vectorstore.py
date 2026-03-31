"""ChromaDB vector store: embed and retrieve clinical note chunks."""
from __future__ import annotations

import functools
from typing import List, Tuple

import numpy as np

from rag_clinical_notes.src.chunking.strategies import Chunk


@functools.lru_cache(maxsize=4)
def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Load and cache a SentenceTransformer model."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def get_or_create_collection(
    persist_directory: str,
    collection_name: str,
    distance_metric: str = "cosine",
):
    """Create a ChromaDB PersistentClient and return (client, collection)."""
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": distance_metric},
    )
    return client, collection


def embed_and_upsert(
    chunks: List[Chunk],
    collection,
    embedding_model,
    batch_size: int = 32,
) -> int:
    """Embed all chunks and upsert into ChromaDB.

    Idempotent: existing chunk_ids are updated in place.
    Returns count of successfully upserted documents.
    """
    total = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.text for c in batch]
        ids = [c.chunk_id for c in batch]
        metadatas = [c.metadata for c in batch]

        embeddings = embedding_model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        embeddings_list = embeddings.tolist()

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings_list,
            metadatas=metadatas,
        )
        total += len(batch)

    return total


def query_collection(
    query_text: str,
    collection,
    embedding_model,
    top_k: int = 20,
) -> List[dict]:
    """Embed query_text and perform ANN search in ChromaDB.

    Returns list of dicts: {"chunk_id", "text", "metadata", "distance"}.
    """
    query_embedding = embedding_model.encode(
        [query_text], normalize_embeddings=True, show_progress_bar=False
    )
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    candidates = []
    for chunk_id, doc, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        candidates.append(
            {
                "chunk_id": chunk_id,
                "text": doc,
                "metadata": metadata,
                "distance": distance,
            }
        )
    return candidates
