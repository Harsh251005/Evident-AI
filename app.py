import streamlit as st
import nltk

nltk.download('stopwords', quiet=True)

from src.pipeline.evident_rag import EvidentAIRAG
from src.retrieval.dynamic_ingest import process_user_upload

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EvidentAI",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Outfit:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:          #0e0d0b;
    --surface:     #161411;
    --surface-2:   #1d1a16;
    --border:      #2a2520;
    --border-2:    #332e28;
    --ink:         #f0ece4;
    --ink-2:       #a8a098;
    --ink-3:       #5a5450;
    --accent:      #d4581a;
    --accent-dim:  rgba(212, 88, 26, 0.12);
    --green:       #3ab870;
    --mono:        'JetBrains Mono', monospace;
    --serif:       'Instrument Serif', Georgia, serif;
    --sans:        'Outfit', sans-serif;
}

/* ── Base ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
    background: var(--bg) !important;
    font-family: var(--sans) !important;
    color: var(--ink) !important;
}

[data-testid="block-container"] {
    padding: 2.8rem 1.5rem 6rem !important;
    max-width: 760px !important;
    margin: 0 auto !important;
}

/* ── Hide chrome ── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; visibility: hidden !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0a0908 !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] [data-testid="block-container"] {
    padding: 2rem 1.2rem 2rem !important;
    max-width: none !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1px dashed var(--border-2) !important;
    border-radius: 10px !important;
    transition: border-color 0.18s !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}

[data-testid="stFileUploader"] section {
    background: transparent !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] > div {
    border-color: var(--accent) transparent transparent transparent !important;
}

/* ── Alerts ── */
.stAlert {
    background: var(--surface) !important;
    border: 1px solid var(--border-2) !important;
    border-radius: 8px !important;
    font-family: var(--sans) !important;
    font-size: 0.82rem !important;
    color: var(--ink-2) !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.5rem 0 !important;
    gap: 0.9rem !important;
}

/* User bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: var(--surface) !important;
    border: 1px solid var(--border-2) !important;
    border-radius: 10px !important;
    padding: 0.9rem 1.1rem !important;
}

/* Assistant bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    border-left: 2px solid var(--accent) !important;
    padding-left: 1.1rem !important;
}

/* Avatar icons */
[data-testid="chatAvatarIcon-user"],
[data-testid="chatAvatarIcon-assistant"] {
    background: var(--surface-2) !important;
    border: 1px solid var(--border-2) !important;
    border-radius: 6px !important;
}

/* Message text */
[data-testid="stChatMessage"] p {
    font-family: var(--sans) !important;
    font-size: 0.95rem !important;
    line-height: 1.75 !important;
    color: var(--ink-2) !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) p {
    color: var(--ink) !important;
}

/* ── Chat input ── */
[data-testid="stChatInputContainer"] {
    background: var(--bg) !important;
    border-top: 1px solid var(--border) !important;
    padding: 0.9rem 0 0 !important;
}

[data-testid="stChatInputContainer"] textarea {
    background: var(--surface) !important;
    border: 1.5px solid var(--border-2) !important;
    border-radius: 10px !important;
    color: var(--ink) !important;
    font-family: var(--sans) !important;
    font-size: 0.95rem !important;
    font-weight: 400 !important;
    transition: border-color 0.18s, box-shadow 0.18s !important;
}

[data-testid="stChatInputContainer"] textarea::placeholder {
    color: var(--ink-3) !important;
    font-weight: 300 !important;
}

[data-testid="stChatInputContainer"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-dim) !important;
    outline: none !important;
}

[data-testid="stChatInputContainer"] button {
    background: var(--accent) !important;
    border: none !important;
    border-radius: 8px !important;
    color: #fff !important;
    transition: filter 0.15s !important;
}

[data-testid="stChatInputContainer"] button:hover {
    filter: brightness(1.12) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-2); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: #443e38; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <style>
    .sidebar-hero {
        font-family: var(--serif);
        font-size: 1.5rem;
        font-style: italic;
        color: var(--ink);
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
        line-height: 1;
    }
    .sidebar-hero span { color: var(--accent); font-style: normal; }
    .sidebar-tagline {
        font-size: 0.72rem;
        color: var(--ink-3);
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 2rem;
        font-family: var(--sans);
    }
    .sidebar-label {
        font-family: var(--mono);
        font-size: 0.65rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--ink-3);
        margin-bottom: 0.8rem;
        display: block;
    }
    </style>
    <div class="sidebar-hero">Evident<span>AI</span></div>
    <div class="sidebar-tagline">Document intelligence</div>
    <span class="sidebar-label">Document</span>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type="pdf",
        label_visibility="collapsed",
    )

    # ── ORIGINAL LOGIC — untouched ──────────────────────────────────────────
    if uploaded_file:
        with st.spinner("Indexing document..."):
            target_collection = process_user_upload(uploaded_file)

        file_size = round(uploaded_file.size / 1024, 1)
        st.markdown(f"""
        <div style="margin-top:1.2rem;background:var(--surface);border:1px solid var(--border-2);
                    border-radius:10px;padding:1rem 1.1rem;">
            <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.6rem;">
                <span style="width:7px;height:7px;border-radius:50%;background:var(--green);
                             display:inline-block;box-shadow:0 0 6px rgba(58,184,112,0.35);
                             flex-shrink:0;"></span>
                <span style="font-family:var(--mono);font-size:0.68rem;
                             color:var(--green);letter-spacing:0.04em;">Ready</span>
            </div>
            <div style="font-family:var(--mono);font-size:0.72rem;color:var(--ink-2);
                        line-height:1.9;word-break:break-word;">
                <div>{uploaded_file.name}</div>
                <div style="color:var(--ink-3);">{file_size} KB</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(
            '<div style="font-family:var(--sans);font-size:0.82rem;color:var(--ink-3);'
            'margin-top:0.5rem;line-height:1.6;">Upload a PDF to begin chatting.</div>',
            unsafe_allow_html=True,
        )
        st.stop()


# ── RAG engine ─────────────────────────────────────────────────────────────────
# ORIGINAL LOGIC — untouched
@st.cache_resource
def get_rag(collection):
    return EvidentAIRAG(collection_name=collection)

rag_engine = get_rag(target_collection)


# ── Page header ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
.page-header {{
    padding-bottom: 1.8rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.6rem;
}}
.page-title {{
    font-family: var(--serif);
    font-size: 2rem;
    font-style: italic;
    color: var(--ink);
    letter-spacing: -0.02em;
    line-height: 1;
    margin-bottom: 0.65rem;
}}
.doc-chip {{
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: var(--surface);
    border: 1px solid var(--border-2);
    border-radius: 6px;
    padding: 0.28rem 0.65rem;
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--ink-3);
}}
.doc-chip .dot {{
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--green);
    display: inline-block;
    box-shadow: 0 0 5px rgba(58,184,112,0.4);
    flex-shrink: 0;
}}
</style>
<div class="page-header">
    <div class="page-title">Ask your document</div>
    <div class="doc-chip">
        <span class="dot"></span>{uploaded_file.name}
    </div>
</div>
""", unsafe_allow_html=True)


# ── Chat ───────────────────────────────────────────────────────────────────────
# ORIGINAL LOGIC — untouched
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center;padding:3.5rem 0 2rem;">
        <div style="font-size:1.8rem;margin-bottom:0.8rem;opacity:0.5;">🔍</div>
        <div style="font-family:var(--sans);font-size:0.9rem;color:var(--ink-3);line-height:1.7;">
            Ask a question about your document.
        </div>
        <div style="font-family:var(--mono);font-size:0.72rem;color:var(--ink-3);
                    opacity:0.55;margin-top:0.3rem;">
            Summaries · specific facts · comparisons
        </div>
    </div>
    """, unsafe_allow_html=True)

if prompt := st.chat_input("Ask anything about your document…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = rag_engine.chain.invoke(prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})