"""Groq chat-completion wrapper with retry logic."""

from __future__ import annotations

import logging
import time

from groq import Groq

from src.config import GROQ_API_KEY, GROQ_MODEL

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def generate(prompt: str, system: str | None = None) -> str:
    """Generate a completion with the configured Groq model."""
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is missing. Add your Groq API key to the .env file "
            "before using the app or evaluation tools."
        )

    client = Groq(api_key=GROQ_API_KEY)
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(1, 5):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.2,
            )
            content = response.choices[0].message.content
            if content is None:
                raise RuntimeError("Groq returned an empty response.")
            return content.strip()
        except Exception as exc:  # pragma: no cover - API failures depend on external service
            status_code = getattr(exc, "status_code", None)
            if attempt >= 4 or not (
                status_code == 429
                or isinstance(status_code, int) and 500 <= status_code < 600
            ):
                raise
            delay = 2 ** (attempt - 1)
            logging.warning("Temporary Groq API error (HTTP %s). Retrying in %s seconds.", status_code, delay)
            time.sleep(delay)

    raise RuntimeError("Generation could not complete after retries.")


if __name__ == "__main__":
    print(generate("Say hi in one word"))
