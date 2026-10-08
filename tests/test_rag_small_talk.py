"""Tests for standalone chat pleasantries in the RAG assistant."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src.rag import NO_ANSWER_MESSAGE, answer


class SmallTalkTests(unittest.TestCase):
    """Check that greetings are friendly without weakening grounded answers."""

    def test_greetings_and_courtesies_return_without_document_lookup(self) -> None:
        """Standalone pleasantries are answered without retrieval or generation."""
        cases = {
            "Hello!": "Hello!",
            "helo": "Hello!",
            "helllo": "Hello!",
            "thankyou": "You're welcome!",
            "thnaks": "You're welcome!",
            "Please.": "Sure!",
            "How are you?": "I'm doing well",
        }
        for prompt, expected in cases.items():
            with self.subTest(prompt=prompt):
                with (
                    patch("src.rag.retrieve") as retrieve,
                    patch("src.rag.retrieve_hybrid") as retrieve_hybrid,
                    patch("src.rag.generate") as generate,
                ):
                    result = answer(prompt, use_hybrid=True)

                self.assertIn(expected, result["answer"])
                self.assertEqual(result["sources"], [])
                self.assertEqual(result["chunks"], [])
                retrieve.assert_not_called()
                retrieve_hybrid.assert_not_called()
                generate.assert_not_called()

    def test_greeting_with_question_still_uses_document_context(self) -> None:
        """A real question following a greeting is not classified as small talk."""
        with patch("src.rag.retrieve", return_value=[] ) as retrieve:
            result = answer("Hello, what is phishing?")

        retrieve.assert_called_once()
        self.assertEqual(result["answer"], NO_ANSWER_MESSAGE)

    def test_typo_in_greeting_with_question_still_uses_document_context(self) -> None:
        """Typo correction must not classify a longer informational request as small talk."""
        with patch("src.rag.retrieve", return_value=[]) as retrieve:
            result = answer("helo, what is phishing?")

        retrieve.assert_called_once()
        self.assertEqual(result["answer"], NO_ANSWER_MESSAGE)


if __name__ == "__main__":
    unittest.main()
