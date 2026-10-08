# Project Spec: CyberRAG — a small, free RAG system over cybersecurity PDFs

> **Instructions for GitHub Copilot (Agent mode):**
> Build this project end to end by following this spec. Work through the build steps **in order**. After each step, run the verification command listed for that step and fix any errors before moving on. Do not ask me questions unless something is truly blocked; make sensible choices and note them in the README. Keep the code simple, readable, and well commented, since a student will present it to a mentor.

---

## 1. Goal

Build a Retrieval-Augmented Generation (RAG) app that answers questions about a small set of cybersecurity PDF documents. Every answer must **cite its sources (filename + page number)**. If the answer is not in the documents, the system must say it doesn't know instead of guessing.

**Hard constraint: everything must be free.** No paid services, no credit card.

---

## 2. Tech stack (fixed, do not substitute)

| Part | Choice |
|---|---|
| Language | Python 3.10+ |
| PDF parsing | `pymupdf` (import as `fitz`) |
| Chunking | `langchain-text-splitters` → `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers`, model `BAAI/bge-small-en-v1.5` (runs locally on CPU, no API key) |
| Vector store | `chromadb` with `PersistentClient`, stored in `./chroma_db` |
| LLM | Groq API via the **`groq`** SDK (`from groq import Groq`) |
| UI | `streamlit` |
| Config | `python-dotenv`, values in `.env` |
| Lexical search (improvement step) | `rank_bm25` |

Use the Groq SDK for text generation.

---

## 3. Project structure

```
cyber-rag/
├── docs/                  # user's PDFs go here (do not commit large PDFs)
├── chroma_db/             # created automatically, git-ignored
├── src/
│   ├── __init__.py
│   ├── config.py          # loads .env, holds constants
│   ├── ingest.py          # parse → clean → chunk → embed → store
│   ├── retriever.py       # query → top-k chunks (vector, then hybrid)
│   ├── llm.py             # Groq call with retry/backoff
│   ├── rag.py             # ties retriever + llm together, builds prompt
│   └── eval.py            # evaluation script
├── app.py                 # Streamlit UI
├── eval_questions.json    # test questions (starter file with placeholders)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 4. Configuration (`.env.example`)

```
GROQ_API_KEY=your_key_here
GROQ_MODEL=allam-2-7b
EMBED_MODEL=BAAI/bge-small-en-v1.5
CHUNK_SIZE=800
CHUNK_OVERLAP=120
TOP_K=5
```

`config.py` must load these with defaults. The model name must be configurable through `.env` because provider model availability can change over time. The README should tell the user to check https://console.groq.com/docs/models if the model name stops working.

`.gitignore` must include: `.env`, `chroma_db/`, `.venv/`, `__pycache__/`, `docs/*.pdf`.

---

## 5. Build steps

### Step 1: Setup
- Create the structure above, `requirements.txt` (pinned loosely, e.g. `pymupdf`, `langchain-text-splitters`, `sentence-transformers`, `chromadb`, `groq`, `streamlit`, `python-dotenv`, `rank_bm25`, `pandas`), `.env.example`, `.gitignore`.
- **Verify:** `pip install -r requirements.txt` succeeds and `python -c "import fitz, chromadb, streamlit"` runs without errors.

### Step 2: Ingestion (`src/ingest.py`)
Pipeline per PDF in `docs/`:
1. **Parse** page by page with PyMuPDF. Keep metadata `{source: filename, page: page_number (1-indexed)}`.
2. **Clean:** collapse repeated whitespace, fix words broken by hyphen + newline, join lines broken mid-sentence, and drop lines that repeat on most pages (headers/footers/page numbers). Skip empty pages.
3. **Chunk** each page's text with `RecursiveCharacterTextSplitter` (`chunk_size` and `chunk_overlap` from config). Each chunk keeps its page's metadata plus a `chunk_id` like `filename_p12_c3`.
4. **Embed** all chunks in batches with the local sentence-transformers model (normalize embeddings).
5. **Store** in a ChromaDB collection named `cyber_docs` using cosine distance. Support a `--reset` CLI flag that deletes and rebuilds the collection.
6. Print a summary: number of PDFs, pages, chunks, and chunks per document.

CLI: `python -m src.ingest --reset`

- **Verify:** put at least one PDF in `docs/`, run the CLI, and confirm the summary prints non-zero chunks.

### Step 3: Retriever (`src/retriever.py`)
- Function `retrieve(query: str, k: int = TOP_K, source: str | None = None) -> list[dict]`.
- Embed the query (for bge models, prefix the query with `"Represent this sentence for searching relevant passages: "`), query Chroma, and return a list of dicts: `{text, source, page, chunk_id, score}`.
- Optional `source` filter restricts results to one document.
- **Verify:** a small `if __name__ == "__main__":` block that runs `retrieve("What is phishing?")` and prints the top results with scores.

### Step 4: LLM wrapper (`src/llm.py`)
- Use the Groq SDK: `client = Groq(api_key=...)` and `client.chat.completions.create(model=..., messages=...)`.
- Function `generate(prompt: str, system: str | None = None) -> str`.
- Set low temperature (0.2).
- Add retry with exponential backoff (up to 4 tries) for rate-limit (429) and temporary 5xx errors.
- If `GROQ_API_KEY` is missing, raise a clear, friendly error message.
- **Verify:** `python -c "from src.llm import generate; print(generate('Say hi in one word'))"`.

### Step 5: RAG pipeline (`src/rag.py`)
- Function `answer(question: str, k: int = TOP_K, source: str | None = None, history: list | None = None) -> dict` returning `{answer, sources, chunks}`.
- Flow: retrieve → build prompt → call LLM → return.
- **System prompt** (use close to this wording):

```
You are a cybersecurity assistant. Answer the question using ONLY the context
provided. After each claim, cite the source in the form [filename, p.N].
If the context does not contain the answer, reply exactly:
"I couldn't find this in the provided documents."
Do not use outside knowledge. Be concise and clear.
```

- The prompt shows each chunk as `[Source: filename, Page: N]` followed by the text, then the question.
- Add a relevance guard: if the best retrieved score is below a configurable threshold, skip the LLM and return the "couldn't find" message. Tune the threshold so it works with cosine similarity.
- Support short chat history (last 3 turns) so follow-up questions work. Keep it simple.
- **Verify:** a `__main__` block that asks one in-scope question and one out-of-scope question (e.g. "Who won the 2018 World Cup?") and prints both results. The out-of-scope one must return the "couldn't find" message.

### Step 6: Streamlit UI (`app.py`)
- Chat interface using `st.chat_message` and `st.chat_input`, with history in `st.session_state`.
- Sidebar: title, a dropdown to filter by document (populated from Chroma metadata, with an "All documents" option), a slider for `top_k` (1–10), and a "Clear chat" button.
- Under each answer, an expander **"Sources"** listing each retrieved chunk: filename, page, similarity score, and the chunk text.
- Show a spinner while generating. Show a friendly error if the API key is missing or the vector store is empty (tell the user to run ingestion first).
- Cache the embedding model and Chroma client with `st.cache_resource`.
- **Verify:** `streamlit run app.py` starts and answers a question end to end.

### Step 7: Evaluation (`src/eval.py`)
- Read `eval_questions.json`, a list of `{question, expected_source, expected_page (optional), expected_keywords: [..]}`. Create the starter file with 3 placeholder examples and a comment in the README telling the user to replace them with 15–20 real questions from their own PDFs.
- For each question compute:
  - **Retrieval hit@k:** whether a chunk from `expected_source` (and `expected_page` if given) appears in the top-k.
  - **Answer keyword score:** fraction of `expected_keywords` that appear in the answer (case-insensitive).
- Include 2–3 out-of-scope questions with `"should_refuse": true` and check that the system refuses.
- Sleep briefly between questions to respect free-tier rate limits.
- Print a results table (use `pandas`) and the overall averages. Save results to `eval_results.csv`.
- CLI: `python -m src.eval`

### Step 8: Improvement, hybrid search (`src/retriever.py`)
- Add `retrieve_hybrid(query, k, source=None)` that combines vector results with BM25 results (`rank_bm25`) using Reciprocal Rank Fusion.
- Build the BM25 index from the chunks stored in Chroma at startup.
- Add a `USE_HYBRID` flag in `.env` (default `false`) and a sidebar checkbox in the UI.
- Make `eval.py` accept `--hybrid` so the user can compare the two modes and report before/after numbers to their mentor.

### Step 9: README (`README.md`)
Include:
1. One-paragraph project description.
2. Architecture diagram (ASCII or Mermaid): `PDFs → clean → chunk → embed → ChromaDB → retrieve → Groq → cited answer`.
3. Setup steps: create venv, install requirements, get a Groq API key at https://console.groq.com/keys, copy `.env.example` to `.env`, add PDFs to `docs/`.
4. Run steps: ingest, launch the app, run the evaluation.
5. A "Design choices" section (why local embeddings, why chunk size 800, why a relevance threshold, why citations).
6. A "Limitations" section (scanned PDFs need OCR, free-tier rate limits, small corpus).
7. A "Results" section with an empty table the user fills in from `eval_results.csv`.

---

## 6. Quality requirements

- Type hints and short docstrings on every function.
- No hardcoded API keys; never print the key.
- Use `logging` instead of bare `print` in library code (CLI summaries may print).
- Handle edge cases: empty `docs/` folder, a PDF with no extractable text (warn and skip), an unreadable PDF (warn and continue).
- Ingestion must be idempotent: re-running without `--reset` should not create duplicate chunks (use deterministic `chunk_id`s with upsert).
- Code must work on Windows, macOS, and Linux (use `pathlib`, no shell-specific commands in code).

---

## 7. Definition of done

- [ ] `python -m src.ingest --reset` indexes the PDFs in `docs/`.
- [ ] `streamlit run app.py` gives cited answers with an expandable Sources panel.
- [ ] An out-of-scope question returns "I couldn't find this in the provided documents."
- [ ] `python -m src.eval` prints hit@k and keyword scores, and saves `eval_results.csv`.
- [ ] `python -m src.eval --hybrid` runs and allows a before/after comparison.
- [ ] README is complete and a new user can follow it from scratch.

---

## 8. Notes for the human (not for Copilot)

1. Create a new empty folder `cyber-rag`, open it in VS Code, and put this file inside it.
2. Open Copilot Chat in **Agent mode** and say: *"Read COPILOT_BUILD_SPEC.md and build the project step by step."*
3. Copy your PDFs into `docs/`, get your API key from Groq, and put it in `.env`.
4. After Copilot finishes, run the app and test it yourself before showing your mentor. Replace the placeholder questions in `eval_questions.json` with real ones from your PDFs.
