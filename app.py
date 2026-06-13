from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.mintri_rag.generation import configured_model, configured_provider, generate_answer
from src.mintri_rag.ingest import ingest_corpus
from src.mintri_rag.paths import CHUNKS_PATH, RAW_DATA_DIR
from src.mintri_rag.retrieval import (
    RETRIEVAL_MODES,
    RAGIndex,
    default_retrieval_mode,
    embedding_model_name,
)
from src.mintri_rag.storage import load_chunks
from src.mintri_rag.text_mining import cluster_topics, extract_keywords, summarize_results


load_dotenv()

st.set_page_config(page_title="CourseLens", page_icon="CL", layout="wide")


@st.cache_resource(show_spinner=False)
def load_index() -> RAGIndex:
    index = RAGIndex.load()
    return index


def reset_index_cache() -> None:
    load_index.clear()


st.title("CourseLens")
st.caption("A RAG assistant for Information Retrieval and Text Mining course material.")

with st.sidebar:
    st.header("Corpus")
    raw_files = sorted(
        p for p in RAW_DATA_DIR.glob("**/*") if p.is_file() and not p.name.startswith(".")
    )
    chunks = load_chunks(CHUNKS_PATH)
    st.metric("Source files", len(raw_files))
    st.metric("Chunks", len(chunks))
    st.metric("LLM", configured_model() if configured_provider() == "gemini" else "Local fallback")
    st.metric("Embeddings", embedding_model_name())

    if raw_files:
        st.write("Documents")
        for file_path in raw_files[:12]:
            st.caption(file_path.relative_to(RAW_DATA_DIR).as_posix())
        if len(raw_files) > 12:
            st.caption(f"...and {len(raw_files) - 12} more")

    if st.button("Rebuild index", use_container_width=True):
        with st.spinner("Ingesting documents and rebuilding the retrieval index..."):
            report = ingest_corpus()
            reset_index_cache()
        st.success(
            f"Indexed {report['chunk_count']} chunks from {report['document_count']} documents."
        )

if not CHUNKS_PATH.exists():
    st.info(
        "Add PDFs, TXT, or Markdown files to data/raw/, then click Rebuild index or run scripts/ingest.py."
    )
else:
    query = st.text_input(
        "Ask a question",
        placeholder="Example: What is the difference between lexical and dense retrieval?",
    )
    col_topk, col_mode = st.columns(2)
    with col_topk:
        top_k = st.slider("Number of retrieved chunks", min_value=3, max_value=10, value=5)
    with col_mode:
        mode_options = list(RETRIEVAL_MODES)
        default_mode = default_retrieval_mode()
        default_index = mode_options.index(default_mode) if default_mode in mode_options else 0
        retrieval_mode = st.selectbox(
            "Retrieval mode",
            options=mode_options,
            index=default_index,
            help=(
                "hybrid = BM25 + dense fused with Reciprocal Rank Fusion. "
                "dense = embeddings only. lexical = BM25 only."
            ),
        )

    if query:
        with st.status("Preparing the answer...", expanded=True) as status:
            try:
                st.write("Loading retrieval index...")
                index = load_index()

                st.write(f"Retrieving with mode: {retrieval_mode}...")
                results = index.search(query, top_k=top_k, mode=retrieval_mode)

                st.write("Generating grounded answer with the configured LLM...")
                answer = generate_answer(query, results)

                status.update(label="Answer ready", state="complete", expanded=False)
            except FileNotFoundError:
                status.update(label="Retrieval index missing", state="error", expanded=True)
                st.warning("The retrieval index is missing. Rebuild the index first.")
                st.stop()

        st.subheader("Answer")
        st.write(answer)

        tab_sources, tab_keywords, tab_summary, tab_topics = st.tabs(
            ["Sources", "Keywords", "Summary", "Topics"]
        )

        with tab_sources:
            rows = [
                {
                    "rank": i + 1,
                    "score": round(item.score, 4),
                    "source": item.chunk["source"],
                    "chunk": item.chunk["chunk_id"],
                    "preview": item.chunk["text"][:320].replace("\n", " "),
                }
                for i, item in enumerate(results)
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            for i, item in enumerate(results, start=1):
                with st.expander(f"[S{i}] {item.chunk['source']} - chunk {item.chunk['chunk_id']}"):
                    st.write(item.chunk["text"])

        with tab_keywords:
            keywords = extract_keywords([item.chunk["text"] for item in results], top_n=12)
            st.write(", ".join(keywords) if keywords else "No keywords found.")

        with tab_summary:
            st.write(summarize_results([item.chunk["text"] for item in results]))

        with tab_topics:
            topic_rows = cluster_topics([item.chunk["text"] for item in results])
            if topic_rows:
                st.dataframe(pd.DataFrame(topic_rows), use_container_width=True)
            else:
                st.write("Not enough retrieved text to cluster.")
