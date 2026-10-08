"""Question-answering pipeline using retrieval + Groq."""

from __future__ import annotations

import re
from typing import Any

from src.config import RELEVANCE_THRESHOLD, TOP_K
from src.llm import generate
from src.retriever import retrieve, retrieve_hybrid

NO_ANSWER_MESSAGE = "I couldn't find this in the provided documents."
SMALL_TALK_RESPONSES = {
    "hello": "Hello! Ask me a cybersecurity question from the provided documents whenever you're ready.",
    "hello there": "Hello! Ask me a cybersecurity question from the provided documents whenever you're ready.",
    "hi": "Hi! Ask me a cybersecurity question from the provided documents whenever you're ready.",
    "hi there": "Hi! Ask me a cybersecurity question from the provided documents whenever you're ready.",
    "hey": "Hey! Ask me a cybersecurity question from the provided documents whenever you're ready.",
    "hey there": "Hey! Ask me a cybersecurity question from the provided documents whenever you're ready.",
    "good morning": "Good morning! Ask me a cybersecurity question from the provided documents whenever you're ready.",
    "good afternoon": "Good afternoon! Ask me a cybersecurity question from the provided documents whenever you're ready.",
    "good evening": "Good evening! Ask me a cybersecurity question from the provided documents whenever you're ready.",
    "thanks": "You're welcome! I'm here if you have another question about the documents.",
    "thank you": "You're welcome! I'm here if you have another question about the documents.",
    "thankyou": "You're welcome! I'm here if you have another question about the documents.",
    "thanks a lot": "You're welcome! I'm here if you have another question about the documents.",
    "thank you so much": "You're welcome! I'm here if you have another question about the documents.",
    "please": "Sure! What would you like to know about the cybersecurity documents?",
    "how are you": "I'm doing well, thanks! Ask me a cybersecurity question from the provided documents whenever you're ready.",
}
SYSTEM_PROMPT = (
    "You are a cybersecurity assistant. Answer the question using ONLY the context "
    "provided. After each claim, cite the source in the form [filename, p.N].\n"
    "If the context does not contain the answer, reply exactly:\n"
    '"I couldn\'t find this in the provided documents."\n'
    "Do not use outside knowledge. Respond in English. Be concise and clear."
)


def _small_talk_response(question: str) -> str | None:
    """Return a friendly response for standalone pleasantries, allowing one typo."""
    normalized = re.sub(r"[^\w\s]", "", question.casefold())
    words = re.sub(r"\s+", " ", normalized).strip().split()
    if not words:
        return None

    normalized = " ".join(words)
    exact_response = SMALL_TALK_RESPONSES.get(normalized)
    if exact_response:
        return exact_response

    for phrase, response in SMALL_TALK_RESPONSES.items():
        target_words = phrase.split()
        if len(words) != len(target_words):
            continue

        typo_count = sum(
            1
            for word, target in zip(words, target_words)
            if word != target and _is_single_typo(word, target)
        )
        if typo_count == 1 and all(
            word == target or _is_single_typo(word, target)
            for word, target in zip(words, target_words)
        ):
            return response

    return None


def _is_single_typo(word: str, target: str) -> bool:
    """Check for one insertion, deletion, substitution, or adjacent transposition."""
    if len(target) < 4 or abs(len(word) - len(target)) > 1 or word == target:
        return False

    word_index = 0
    target_index = 0
    edits = 0
    while word_index < len(word) and target_index < len(target):
        if word[word_index] == target[target_index]:
            word_index += 1
            target_index += 1
            continue

        edits += 1
        if edits > 1:
            return False
        if (
            word_index + 1 < len(word)
            and target_index + 1 < len(target)
            and word[word_index] == target[target_index + 1]
            and word[word_index + 1] == target[target_index]
        ):
            word_index += 2
            target_index += 2
        elif len(word) > len(target):
            word_index += 1
        elif len(target) > len(word):
            target_index += 1
        else:
            word_index += 1
            target_index += 1

    if word_index < len(word) or target_index < len(target):
        edits += 1
    return edits == 1


def _format_history(history: list[dict[str, str]] | list[str] | None) -> str:
    """Translate recent chat history into plain-text context."""
    if not history:
        return ""

    recent = history[-3:]
    lines: list[str] = []
    for item in recent:
        if isinstance(item, dict):
            if item.get("role") == "user":
                lines.append(f"User: {item.get('content', '')}")
            elif item.get("role") == "assistant":
                lines.append(f"Assistant: {item.get('content', '')}")
        elif isinstance(item, str):
            lines.append(item)
    return "\n".join(lines)


def _build_context(chunks: list[dict[str, Any]]) -> str:
    """Render the retrieved chunks for the prompt."""
    context_blocks: list[str] = []
    for chunk in chunks:
        context_blocks.append(
            f"[Source: {chunk['source']}, Page: {chunk['page']}]\n{chunk['text']}"
        )
    return "\n\n".join(context_blocks)


def answer(
    question: str,
    k: int = TOP_K,
    source: str | None = None,
    history: list | None = None,
    use_hybrid: bool = False,
) -> dict[str, Any]:
    """Answer a question using retrieved context and the Groq LLM."""
    if not question or not question.strip():
        return {"answer": NO_ANSWER_MESSAGE, "sources": [], "chunks": []}

    small_talk = _small_talk_response(question)
    if small_talk:
        return {"answer": small_talk, "sources": [], "chunks": []}

    chunks = retrieve_hybrid(question, k=k, source=source) if use_hybrid else retrieve(question, k=k, source=source)
    if not chunks:
        return {"answer": NO_ANSWER_MESSAGE, "sources": [], "chunks": []}

    if use_hybrid:
        # RRF scores rank the two retrievers and are intentionally much smaller
        # than cosine similarity scores, so use the vector relevance signal.
        relevance_score = max(chunk["query_similarity"] for chunk in chunks)
    else:
        relevance_score = chunks[0]["score"]

    if relevance_score < RELEVANCE_THRESHOLD:
        return {"answer": NO_ANSWER_MESSAGE, "sources": [], "chunks": chunks}

    context = _build_context(chunks)
    history_text = _format_history(history)
    prompt = (
        f"Context:\n{context}\n\n"
        f"Recent conversation:\n{history_text}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    llm_answer = generate(prompt, system=SYSTEM_PROMPT)
    sources = [
        {
            "source": chunk["source"],
            "page": chunk["page"],
            "score": chunk["score"],
            **(
                {
                    "similarity": chunk["similarity"],
                    "bm25_score": chunk["bm25_score"],
                }
                if use_hybrid
                else {}
            ),
        }
        for chunk in chunks
    ]
    return {"answer": llm_answer.strip(), "sources": sources, "chunks": chunks}


if __name__ == "__main__":
    in_scope = answer("What is phishing?", k=3)
    out_of_scope = answer("Who won the 2018 World Cup?", k=3)
    print("In scope:", in_scope["answer"])
    print("Out of scope:", out_of_scope["answer"])
