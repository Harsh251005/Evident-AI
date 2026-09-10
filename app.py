import tempfile
from pathlib import Path

import streamlit as st

from src.pipeline.ingestion import ingest_document
from src.retrieval.retriever import retrieve
from src.generation.generator import generate_answer

st.set_page_config(
    page_title="EvidentAI",
    page_icon="🔍",
    layout="centered",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.app-header {
    text-align: center;
    padding: 1.6rem 0 1.2rem;
    border-bottom: 1px solid rgba(128,128,128,0.2);
    margin-bottom: 1.4rem;
}
.app-header h1 {
    font-family: 'DM Mono', monospace;
    font-size: 1.5rem;
    font-weight: 500;
    letter-spacing: -0.02em;
    margin: 0;
}
.app-header p {
    font-size: 0.8rem;
    opacity: 0.6;
    margin: 0.3rem 0 0;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.doc-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    border: 1px solid rgba(128,128,128,0.3);
    border-radius: 6px;
    padding: 0.4rem 0.8rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    opacity: 0.75;
    margin-bottom: 1.2rem;
    width: fit-content;
}
.doc-badge span.dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #4ade80;
    display: inline-block;
}
.source-chip {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    opacity: 0.6;
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 4px;
    padding: 0.15rem 0.5rem;
    margin: 0.15rem 0.25rem 0 0;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="app-header">
    <h1>🔍 EvidentAI</h1>
    <p>Hybrid Retrieval · Reranking · Grounded Answer Generation</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("**Document**")
    uploaded_file = st.file_uploader(
        "Upload a PDF to begin", type="pdf", label_visibility="collapsed"
    )

    st.markdown("**Retrieval settings**")
    retrieval_mode = st.selectbox(
        "Mode", ["hybrid", "vector", "bm25"], index=0
    )
    use_rerank = st.checkbox("Rerank with cross-encoder", value=True)
    show_sources = st.checkbox("Show retrieved chunks", value=False)

    if not uploaded_file:
        st.markdown(
            "<div style='font-size:0.8rem; opacity:0.6; margin-top:0.5rem;'>"
            "Upload a PDF to start chatting.</div>",
            unsafe_allow_html=True,
        )
        st.stop()


@st.cache_resource(show_spinner=False)
def get_collection(file_bytes: bytes, file_name: str) -> str:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / file_name
        tmp_path.write_bytes(file_bytes)
        return ingest_document(str(tmp_path))


with st.spinner("Indexing document..."):
    collection_name = get_collection(uploaded_file.getvalue(), uploaded_file.name)

st.markdown(
    f"""
<div class="doc-badge">
    <span class="dot"></span>{uploaded_file.name}
</div>
""",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask anything about your document…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving + generating..."):
            # Retrieve once and reuse it for both generation and the
            # sources panel below — a separate second retrieve() call here
            # used to double the retrieval + reranking cost per question.
            docs = retrieve(
                query=prompt,
                collection_name=collection_name,
                mode=retrieval_mode,
                rerank=use_rerank,
            )
            answer = generate_answer(query=prompt, context_docs=docs)
            st.markdown(answer)

            if show_sources:
                with st.expander(f"Retrieved chunks ({len(docs)})"):
                    for doc in docs:
                        page = doc.metadata.get("page_no", "?")
                        score = doc.metadata.get(
                            "rerank_score", doc.metadata.get("score")
                        )
                        score_str = f"{score:.3f}" if score is not None else "n/a"
                        st.markdown(
                            f"<span class='source-chip'>Page {page} · "
                            f"score {score_str}</span>",
                            unsafe_allow_html=True,
                        )
                        st.caption(doc.page_content[:400])

    st.session_state.messages.append({"role": "assistant", "content": answer})
