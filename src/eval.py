"""Evaluate retrieval and answer quality for the CyberRAG project."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.config import TOP_K
from src.rag import NO_ANSWER_MESSAGE, answer
from src.retriever import retrieve, retrieve_hybrid

BASE_DIR = Path(__file__).resolve().parent.parent


def evaluate(use_hybrid: bool = False, top_k: int = TOP_K) -> pd.DataFrame:
    """Run retrieval and answer checks across the benchmark questions."""
    questions_path = BASE_DIR / "eval_questions.json"
    with questions_path.open("r", encoding="utf-8") as handle:
        questions = json.load(handle)

    rows: list[dict[str, object]] = []
    for item in questions:
        question = item.get("question", "")
        expected_source = item.get("expected_source")
        expected_page = item.get("expected_page")
        expected_keywords = item.get("expected_keywords", [])
        should_refuse = bool(item.get("should_refuse", False))

        if use_hybrid:
            chunks = retrieve_hybrid(question, k=top_k, source=expected_source)
        else:
            chunks = retrieve(question, k=top_k, source=expected_source)

        if expected_source:
            hit_source = any(chunk["source"] == expected_source for chunk in chunks)
        else:
            hit_source = False

        if expected_page:
            hit_page = any(
                chunk["source"] == expected_source and chunk["page"] == expected_page for chunk in chunks
            )
        else:
            hit_page = hit_source

        result = answer(question, k=top_k, source=expected_source, use_hybrid=use_hybrid)
        answer_text = result["answer"]
        answer_keywords = expected_keywords or []
        keyword_matches = sum(1 for keyword in answer_keywords if keyword.lower() in answer_text.lower())
        keyword_score = (keyword_matches / len(answer_keywords)) if answer_keywords else 1.0

        if should_refuse:
            refusal_ok = NO_ANSWER_MESSAGE in answer_text
        else:
            refusal_ok = True

        rows.append(
            {
                "question": question,
                "expected_source": expected_source,
                "expected_page": expected_page,
                "retrieval_hit_at_k": hit_page,
                "keyword_score": keyword_score,
                "answer": answer_text,
                "refusal_ok": refusal_ok,
                "should_refuse": should_refuse,
            }
        )
        time.sleep(0.4)

    df = pd.DataFrame(rows)
    return df


def main() -> None:
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate the CyberRAG retrieval and answer quality.")
    parser.add_argument("--hybrid", action="store_true", help="Use the hybrid retriever for the evaluation run.")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Number of retrieved chunks to inspect for each question.")
    args = parser.parse_args()

    dataframe = evaluate(use_hybrid=args.hybrid, top_k=args.top_k)
    print(dataframe.to_string(index=False))

    averages = {
        "retrieval_hit_at_k": dataframe["retrieval_hit_at_k"].mean(),
        "keyword_score": dataframe["keyword_score"].mean(),
    }
    print("\nAverages:")
    print(pd.DataFrame([averages]).to_string(index=False))

    csv_path = BASE_DIR / "eval_results.csv"
    dataframe.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")


if __name__ == "__main__":
    main()
