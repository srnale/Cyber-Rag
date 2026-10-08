# CyberRAG

CyberRAG is a small, free, local retrieval-augmented generation project for answering questions about cybersecurity PDFs. It parses PDF content, cleans and chunks it, embeds the chunks with a local sentence-transformers model, stores them in ChromaDB, and answers questions with citations to the originating document and page. It refuses to guess when the answer is not present in the user-provided PDF corpus.

## Architecture

```text
PDFs → clean → chunk → embed → ChromaDB → retrieve → Groq → cited answer
```

## Setup

1. Create and activate a virtual environment.

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install the project requirements.

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Create a local `.env` file from `.env.example`.

   ```bash
   copy .env.example .env
   ```

4. Add your Groq API key to `.env`. The default `GROQ_MODEL` is `allam-2-7b`, selected as the smallest active general chat model returned for this account during verification. Its 7B size makes it the lightest listed choice (and likely the lowest cost), but check current per-token pricing and model access in your Groq account.
   - Get an API key from: https://console.groq.com/keys
   - Check currently available models at: https://console.groq.com/docs/models

5. Add your PDF files to the `docs/` directory. The project ignores `docs/*.pdf` in `.gitignore`.

## Run

### Ingest PDFs

```bash
python -m src.ingest --reset
```

### Launch the web app

```bash
streamlit run app.py
```

### Evaluate the system

```bash
python -m src.eval
python -m src.eval --hybrid
```

## Design choices

- Local embeddings: the project uses `BAAI/bge-small-en-v1.5` so it runs without paying for external vector APIs.
- Lightweight LLM: `allam-2-7b` is the smallest active general chat model returned by this Groq account's model list. A live English generation request succeeded; model availability and pricing can vary by account. The system prompt requests English responses for this English-language project.
- Chunk size 800: this balances context length with preserving meaningful cybersecurity passages.
- Relevance threshold: if the best similarity score is too low, the app refuses to answer instead of hallucinating.
- Hybrid search: BM25 and vector ranks are combined with Reciprocal Rank Fusion (RRF). RRF scores are ranking values, not cosine similarities, so the relevance guard uses the separate vector-similarity signal; the Sources panel shows both values.
- Citations: every answer must cite the source filename and page number so the user can verify the claim in the original PDF.
- Basic chat: standalone greetings and courtesy messages (for example, "hello", "thank you", and "please") receive a short friendly reply without document retrieval. A single typo in a standalone pleasantry is tolerated; questions and other informational requests still require support from the provided documents.

## Limitations

- Scanned PDFs may need OCR before they work cleanly.
- Groq API usage may be rate limited; the app includes retry/backoff logic and the evaluation script pauses briefly between questions.
- This is intended for a small corpus, not a massive knowledge base.


