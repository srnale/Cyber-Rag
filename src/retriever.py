"""Query the local ChromaDB vector store and hybrid-search index."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL, TOP_K


def get_collection() -> Any:
    """Return the persistent Chroma collection."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Get a cached local embedding model."""
    return SentenceTransformer(EMBED_MODEL)


def _build_query_embedding(query: str) -> list[float]:
    """Embed a search query with the BGE retrieval prefix used in search tasks."""
    model = get_embedder()
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    embedding = model.encode(prefixed, normalize_embeddings=True, convert_to_numpy=True)
    return embedding.tolist()


def retrieve(query: str, k: int = TOP_K, source: str | None = None) -> list[dict[str, Any]]:
    """Return the top-k vector-search results for a query."""
    collection = get_collection()
    if not query or not query.strip():
        return []

    where = {"source": source} if source else None
    results = collection.query(
        query_embeddings=[_build_query_embedding(query)],
        n_results=k,
        include=["documents", "metadatas", "distances"],
        where=where,
    )

    output: list[dict[str, Any]] = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for document, metadata, distance in zip(documents, metadatas, distances):
        if metadata is None:
            continue
        score = max(0.0, 1.0 - float(distance))
        output.append(
            {
                "text": document,
                "source": metadata.get("source", "unknown"),
                "page": int(metadata.get("page", 1)),
                "chunk_id": metadata.get("chunk_id", ""),
                "score": round(score, 4),
            }
        )
    return output


def _bm25_records(source: str | None = None) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Build BM25 records from the existing Chroma collection."""
    collection = get_collection()
    result = collection.get(include=["documents", "metadatas"], where={"source": source} if source else None)
    documents = result.get("documents", []) or []
    metadatas = result.get("metadatas", []) or []
    ids = result.get("ids", []) or []

    records: list[dict[str, Any]] = []
    for doc_id, document, metadata in zip(ids, documents, metadatas):
        if not document:
            continue
        records.append(
            {
                "chunk_id": metadata.get("chunk_id", doc_id) if metadata else doc_id,
                "text": document,
                "source": metadata.get("source", "unknown") if metadata else "unknown",
                "page": int(metadata.get("page", 1)) if metadata else 1,
            }
        )
    return [item["text"] for item in records], [item["chunk_id"] for item in records], records


def retrieve_hybrid(query: str, k: int = TOP_K, source: str | None = None) -> list[dict[str, Any]]:
    """Fuse vector and BM25 results with reciprocal rank fusion."""
    if not query or not query.strip():
        return []

    vector_results = retrieve(query=query, k=k * 3, source=source)
    bm25_documents, bm25_ids, bm25_records = _bm25_records(source=source)
    if not bm25_documents:
        query_similarity = max((result["score"] for result in vector_results), default=0.0)
        return [
            {
                **result,
                "similarity": result["score"],
                "bm25_score": 0.0,
                "query_similarity": query_similarity,
            }
            for result in vector_results[:k]
        ]

    tokenized = [doc.lower().split() for doc in bm25_documents]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(query.lower().split())
    scored_records = sorted(
        zip(bm25_ids, bm25_scores, bm25_records),
        key=lambda item: item[1],
        reverse=True,
    )
    bm25_score_by_id = {
        record["chunk_id"]: float(score)
        for _, score, record in scored_records
    }
    ranked = scored_records[: k * 3]

    rrf_scores: dict[str, float] = {}
    vector_scores = {result["chunk_id"]: result["score"] for result in vector_results}
    query_similarity = max(vector_scores.values(), default=0.0)

    for rank, result in enumerate(vector_results, start=1):
        key = result["chunk_id"]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (60 + rank)

    for rank, (_, _, record) in enumerate(ranked, start=1):
        key = record["chunk_id"]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (60 + rank)

    combined: list[dict[str, Any]] = []
    seen: set[str] = set()
    for result in vector_results:
        key = result["chunk_id"]
        if key in seen:
            continue
        seen.add(key)
        combined.append(
            {
                "text": result["text"],
                "source": result["source"],
                "page": result["page"],
                "chunk_id": key,
                "score": round(rrf_scores.get(key, 0.0), 4),
                "similarity": result["score"],
                "bm25_score": bm25_score_by_id.get(key, 0.0),
                "query_similarity": query_similarity,
            }
        )

    for _, _, record in ranked:
        key = record["chunk_id"]
        if key in seen:
            continue
        seen.add(key)
        combined.append(
            {
                "text": record["text"],
                "source": record["source"],
                "page": record["page"],
                "chunk_id": key,
                "score": round(rrf_scores.get(key, 0.0), 4),
                "similarity": 0.0,
                "bm25_score": bm25_score_by_id.get(key, 0.0),
                "query_similarity": query_similarity,
            }
        )

    combined = sorted(combined, key=lambda item: item["score"], reverse=True)[:k]
    return combined


if __name__ == "__main__":
    results = retrieve("What is phishing?")
    for result in results[:5]:
        print(f"{result['source']} | page {result['page']} | score={result['score']}: {result['text'][:180]}")
