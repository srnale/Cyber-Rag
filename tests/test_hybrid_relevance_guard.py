"""Regression tests for hybrid RAG relevance gating."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src.rag import NO_ANSWER_MESSAGE, answer


class HybridRelevanceGuardTests(unittest.TestCase):
    """Ensure the RAG relevance check uses cosine similarity, not RRF rank."""

    def test_relevant_hybrid_result_reaches_llm_despite_small_rrf_score(self) -> None:
        """A valid hybrid result has a small RRF score but relevant vector score."""
        chunks = [
            {
                "text": "Phishing steals credentials.",
                "source": "security.pdf",
                "page": 4,
                "chunk_id": "security_p4_c1",
                "score": 0.03,
                "similarity": 0.81,
                "bm25_score": 2.4,
                "query_similarity": 0.81,
            }
        ]
        with (
            patch("src.rag.retrieve_hybrid", return_value=chunks),
            patch("src.rag.generate", return_value="Phishing steals credentials [security.pdf, p.4].") as generate,
        ):
            result = answer("What is phishing?", use_hybrid=True)

        self.assertNotEqual(result["answer"], NO_ANSWER_MESSAGE)
        generate.assert_called_once()

    def test_irrelevant_hybrid_result_is_still_rejected(self) -> None:
        """A low vector similarity continues to trigger the refusal guard."""
        chunks = [
            {
                "text": "Unrelated passage.",
                "source": "security.pdf",
                "page": 1,
                "chunk_id": "security_p1_c1",
                "score": 0.03,
                "similarity": 0.12,
                "bm25_score": 0.2,
                "query_similarity": 0.12,
            }
        ]
        with (
            patch("src.rag.retrieve_hybrid", return_value=chunks),
            patch("src.rag.generate") as generate,
        ):
            result = answer("An unrelated question", use_hybrid=True)

        self.assertEqual(result["answer"], NO_ANSWER_MESSAGE)
        generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
