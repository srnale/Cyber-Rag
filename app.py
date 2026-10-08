"""Streamlit UI for the CyberRAG application."""

from __future__ import annotations

from typing import Any

import streamlit as st
from sentence_transformers import SentenceTransformer

from src.config import EMBED_MODEL, GROQ_API_KEY, USE_HYBRID
from src.rag import NO_ANSWER_MESSAGE, answer
from src.retriever import get_collection

st.set_page_config(page_title="CyberRAG", page_icon="🛡️")


@st.cache_resource
def get_embedder() -> SentenceTransformer:
    """Share one local embedding model instance across the app."""
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource
def get_chroma_collection() -> Any:
    """Share one Chroma client/collection across the app."""
    return get_collection()


def _source_options() -> list[str]:
    """List documents currently indexed in Chroma for filtering."""
    collection = get_chroma_collection()
    try:
        result = collection.get(include=["metadatas"], limit=10000)
    except Exception:
        return ["All documents"]
    metadatas = result.get("metadatas", []) or []
    sources = sorted({metadata.get("source") for metadata in metadatas if isinstance(metadata, dict) and metadata.get("source")})
    return ["All documents", *sources]


if "messages" not in st.session_state:
    st.session_state.messages = []

st.sidebar.title("CyberRAG")
selected_source = st.sidebar.selectbox("Filter by document", _source_options())
selected_top_k = st.sidebar.slider("Top K", 1, 10, 5)
use_hybrid = st.sidebar.checkbox("Use hybrid retrieval", value=USE_HYBRID)

if st.sidebar.button("Clear chat"):
    st.session_state.messages = []

if not GROQ_API_KEY:
    st.error("Missing Groq API key. Add GROQ_API_KEY to your .env file before running the app.")
    st.stop()

collection = get_chroma_collection()
try:
    doc_count = collection.count()
except Exception:
    doc_count = 0

if doc_count == 0:
    st.warning("The vector store is empty. Run ingestion first with `python -m src.ingest --reset`.")
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    if "similarity" in source:
                        st.write(
                            f"- {source['source']} | page {source['page']} | "
                            f"RRF {source['score']:.4f} | vector similarity {source['similarity']:.4f}"
                        )
                    else:
                        st.write(f"- {source['source']} | page {source['page']} | similarity {source['score']:.4f}")

prompt = st.chat_input("Ask a cybersecurity question")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = [
        {"role": item["role"], "content": item["content"]}
        for item in st.session_state.messages[-6:]
    ]
    with st.spinner("Searching the documents and generating an answer..."):
        response = answer(
            prompt,
            k=selected_top_k,
            source=None if selected_source == "All documents" else selected_source,
            history=history,
            use_hybrid=use_hybrid,
        )

    assistant_message = {
        "role": "assistant",
        "content": response["answer"],
        "sources": response.get("sources", []),
    }
    st.session_state.messages.append(assistant_message)

    with st.chat_message("assistant"):
        st.markdown(response["answer"])
        if response.get("sources"):
            with st.expander("Sources"):
                for item in response["sources"]:
                    if "similarity" in item:
                        st.write(
                            f"- {item['source']} | page {item['page']} | "
                            f"RRF {item['score']:.4f} | vector similarity {item['similarity']:.4f}"
                        )
                    else:
                        st.write(f"- {item['source']} | page {item['page']} | similarity {item['score']:.4f}")
        elif response["answer"] == NO_ANSWER_MESSAGE:
            st.caption("No relevant document chunks were retrieved.")
