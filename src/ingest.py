"""Ingest cybersecurity PDFs into a local ChromaDB collection."""

from __future__ import annotations

import argparse
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import chromadb
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from src.config import CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION_NAME, DOCS_DIR, EMBED_MODEL

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def clean_pdf_text(text: str) -> str:
    """Normalize raw PDF text to a readable, chunk-friendly format."""
    if not text:
        return ""

    text = text.replace("\r", "\n")
    text = re.sub(r"(?<=\w)-\s*\n\s*", "", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    lines = []
    for raw_line in text.splitlines():
        cleaned_line = re.sub(r"\s+", " ", raw_line).strip()
        if cleaned_line:
            lines.append(cleaned_line)

    merged: list[str] = []
    for line in lines:
        if not merged:
            merged.append(line)
            continue

        previous = merged[-1]
        if previous.endswith(("-", "—")):
            merged[-1] = f"{previous.rstrip('- ')}{line}"
        elif len(previous) < 40 and line and line[0].islower() and not previous.endswith((".", "!", "?", ";", ":")):
            merged[-1] = f"{previous} {line}"
        else:
            merged.append(line)

    counts = Counter(line for line in merged)
    repeated_lines = {
        line
        for line, count in counts.items()
        if count >= max(2, len(merged) // 3) and len(line) < 80
    }

    filtered: list[str] = []
    for line in merged:
        if line in repeated_lines:
            continue
        if re.fullmatch(r"(?:page\s*\d+|\d+|[A-Za-z]+)", line, flags=re.IGNORECASE):
            continue
        filtered.append(line)

    return "\n".join(filtered).strip()


def iter_pdf_chunks() -> tuple[list[Path], list[dict[str, Any]], int, int]:
    """Collect text chunks from all PDFs in docs/."""
    pdf_paths = sorted(DOCS_DIR.glob("*.pdf"))
    if not DOCS_DIR.exists():
        DOCS_DIR.mkdir(parents=True, exist_ok=True)

    chunk_records: list[dict[str, Any]] = []
    page_count = 0

    for pdf_path in pdf_paths:
        try:
            document = fitz.open(pdf_path)
        except Exception as exc:  # pragma: no cover - runtime PDF failure
            logging.warning("Skipping unreadable PDF %s: %s", pdf_path.name, exc)
            continue

        for page_number in range(document.page_count):
            page = document[page_number]
            raw_text = page.get_text("text")
            if not raw_text or not raw_text.strip():
                continue

            cleaned_text = clean_pdf_text(raw_text)
            if not cleaned_text:
                continue

            page_count += 1
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", " ", ""],
            )
            chunks = splitter.split_text(cleaned_text)
            for chunk_index, chunk in enumerate(chunks, start=1):
                chunk_text = chunk.strip()
                if not chunk_text:
                    continue
                chunk_id = f"{pdf_path.stem}_p{page_number + 1}_c{chunk_index}"
                chunk_records.append(
                    {
                        "id": chunk_id,
                        "text": chunk_text,
                        "metadata": {
                            "source": pdf_path.name,
                            "page": page_number + 1,
                            "chunk_id": chunk_id,
                        },
                    }
                )

        document.close()

    return pdf_paths, chunk_records, len(pdf_paths), page_count


def ingest_documents(reset: bool = False) -> dict[str, Any]:
    """Parse, clean, embed, and store all PDF chunks."""
    pdf_paths, chunk_records, pdf_count, page_count = iter_pdf_chunks()

    if not pdf_paths:
        logging.warning("No PDFs found in %s. Add PDFs to the docs folder before running ingestion.", DOCS_DIR)
        return {
            "pdf_count": 0,
            "page_count": 0,
            "chunk_count": 0,
            "chunks_per_document": {},
        }

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection_names = [item.name for item in client.list_collections()]
        if reset and COLLECTION_NAME in collection_names:
            client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    if not chunk_records:
        logging.warning("No extractable text found in the PDFs in %s.", DOCS_DIR)
        return {
            "pdf_count": pdf_count,
            "page_count": page_count,
            "chunk_count": 0,
            "chunks_per_document": {},
        }

    model = SentenceTransformer(EMBED_MODEL)
    texts = [item["text"] for item in chunk_records]
    batch_size = 32
    all_embeddings: list[list[float]] = []
    for index in range(0, len(texts), batch_size):
        embeddings = model.encode(
            texts[index : index + batch_size],
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        all_embeddings.extend(embeddings.tolist())

    ids = [item["id"] for item in chunk_records]
    collection.upsert(
        ids=ids,
        embeddings=all_embeddings,
        metadatas=[item["metadata"] for item in chunk_records],
        documents=texts,
    )

    chunks_by_document: dict[str, int] = {}
    for item in chunk_records:
        filename = item["metadata"]["source"]
        chunks_by_document[filename] = chunks_by_document.get(filename, 0) + 1

    summary = {
        "pdf_count": pdf_count,
        "page_count": page_count,
        "chunk_count": len(chunk_records),
        "chunks_per_document": chunks_by_document,
    }

    print(f"Indexed {summary['pdf_count']} PDFs, {summary['page_count']} pages, {summary['chunk_count']} chunks.")
    for filename, count in chunks_by_document.items():
        print(f"  {filename}: {count} chunks")

    return summary


def main() -> None:
    """CLI entry point for ingestion."""
    parser = argparse.ArgumentParser(description="Index cybersecurity PDFs into ChromaDB.")
    parser.add_argument("--reset", action="store_true", help="Delete and rebuild the Chroma collection.")
    args = parser.parse_args()
    ingest_documents(reset=args.reset)


if __name__ == "__main__":
    main()
