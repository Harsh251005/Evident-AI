import streamlit as st
import nltk

nltk.download('stopwords', quiet=True)

from src.pipeline.evident_rag import EvidentAIRAG
from src.retrieval.dynamic_ingest import process_user_upload

st.set_page_config(
    page_title="EvidentAI",
    page_icon="🔍",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0d0f;
    color: #e2e2e6;
}

.stApp { background-color: #0d0d0f; }

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Page header ── */
.app-header {
    text-align: center;
    padding: 2.2rem 0 1.4rem;
    border-bottom: 1px solid #1e1e24;
    margin-bottom: 1.6rem;
}
.app-header h1 {
    font-family: 'DM Mono', monospace;
    font-size: 1.55rem;
    font-weight: 500;
    letter-spacing: -0.02em;
    color: #f0f0f5;
    margin: 0;
}
.app-header p {
    font-size: 0.8rem;
    color: #5a5a72;
    margin: 0.3rem 0 0;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── Doc badge shown after upload ── */
.doc-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: #151519;
    border: 1px solid #2a2a38;
    border-radius: 6px;
    padding: 0.45rem 0.85rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #7c7caa;
    margin-bottom: 1.4rem;
    width: fit-content;
}
.doc-badge span.dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #4ade80;
    display: inline-block;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0a0a0c !important;
    border-right: 1px solid #1a1a22 !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.4rem 1rem; }

.sidebar-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #44445a;
    margin-bottom: 0.6rem;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #111116 !important;
    border: 1px dashed #2a2a3a !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #4a4a7a !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.6rem 0 !important;
}

/* User bubble */
[data-testid="stChatMessage"][data-testid*="user"],
.stChatMessage:has([data-testid="chatAvatarIcon-user"]) {
    background: #13131a !important;
    border-radius: 10px !important;
    border: 1px solid #1e1e2a !important;
    padding: 0.8rem 1rem !important;
}

/* Assistant bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    border-left: 2px solid #3b3b6a;
    padding-left: 1rem !important;
}

/* ── Chat input ── */
[data-testid="stChatInputContainer"] {
    border-top: 1px solid #1a1a22 !important;
    background: #0d0d0f !important;
    padding-top: 0.8rem;
}
[data-testid="stChatInputContainer"] textarea {
    background: #111116 !important;
    border: 1px solid #2a2a3a !important;
    border-radius: 8px !important;
    color: #e2e2e6 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
}
[data-testid="stChatInputContainer"] textarea:focus {
    border-color: #4a4a8a !important;
    box-shadow: 0 0 0 2px rgba(74,74,138,0.15) !important;
}

/* ── Spinner / info / success ── */
.stAlert {
    background: #111116 !important;
    border: 1px solid #1e1e2a !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d0d0f; }
::-webkit-scrollbar-thumb { background: #2a2a3a; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("""
<div class="app-header">
    <h1>🔍 EvidentAI</h1>
    <p>Personal Document Intelligence</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.markdown('<div class="sidebar-label">Document</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a PDF to begin", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        with st.spinner("Indexing document..."):
            target_collection = process_user_upload(uploaded_file)
        st.success(f"Ready: {uploaded_file.name}")
        st.markdown(f"""
        <div style="margin-top:1rem; font-family:'DM Mono',monospace; font-size:0.68rem; color:#44445a; line-height:1.8;">
            <div>📄 {uploaded_file.name}</div>
            <div>{round(uploaded_file.size/1024, 1)} KB</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-size:0.8rem; color:#44445a; margin-top:0.5rem;">Upload a PDF to start chatting.</div>', unsafe_allow_html=True)
        st.stop()

# ── RAG Engine ──
@st.cache_resource
def get_rag(collection):
    return EvidentAIRAG(collection_name=collection)

rag_engine = get_rag(target_collection)

# ── Active doc badge ──
st.markdown(f"""
<div class="doc-badge">
    <span class="dot"></span>{uploaded_file.name}
</div>
""", unsafe_allow_html=True)

# ── Chat Interface ──
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
        response = rag_engine.chain.invoke(prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})