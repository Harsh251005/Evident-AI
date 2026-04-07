import streamlit as st
import os
from src.pipeline.ingestion import ingestion_pipeline
from src.pipeline.retrieval import setup_bm25
from src.retrieval.retriever import retrieve
from src.generation.llm import generate_answer
from src.generation.prompt import build_prompt
from src.ingestion.vector_store import generate_collection_name
from config import settings

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="EvidentAI | Document Intelligence",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stChatItem {
        border-radius: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🔍 EvidentAI")
st.caption("Ask questions and get evidence-backed answers from your documents.")
st.markdown("---")

# --- SESSION STATE INITIALIZATION ---
# This ensures data persists across Streamlit's "re-runs"
if "processed_pdf" not in st.session_state:
    st.session_state.processed_pdf = None
if "bm25" not in st.session_state:
    st.session_state.bm25 = None
if "collection_name" not in st.session_state:
    st.session_state.collection_name = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR: UPLOAD ---
with st.sidebar:
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF to begin", type="pdf")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.info("System is using optimized RAG settings from config.")

# --- PROCESSING LOGIC ---
if uploaded_file:
    # 1. Save uploaded file to local directory
    pdf_path = os.path.join("data", "sample_pdf", uploaded_file.name)
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 2. Trigger Ingestion only if it's a new file
    if st.session_state.processed_pdf != uploaded_file.name:
        with st.status(f"Analyzing {uploaded_file.name}...", expanded=True) as status:
            # Run the modular pipeline
            doc, chunks, embeds = ingestion_pipeline(pdf_path)

            # Generate/Get Collection Name
            coll_name = generate_collection_name(pdf_path)

            # Setup/Load BM25 Index
            bm25 = setup_bm25(
                coll_name,
                chunks,
                [c.metadata for c in chunks] if chunks else None
            )

            # Update Session State
            st.session_state.processed_pdf = uploaded_file.name
            st.session_state.collection_name = coll_name
            st.session_state.bm25 = bm25

            status.update(label="Analysis Complete! ✅", state="complete", expanded=False)

# --- CHAT INTERFACE ---
# Only show the chat if a document has been successfully processed
if st.session_state.processed_pdf:

    # Display existing chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle new user input
    if query := st.chat_input("What would you like to know?"):

        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Searching through pages..."):
                # 1. Retrieve relevant chunks
                context = retrieve(
                    query,
                    st.session_state.collection_name,
                    bm25=st.session_state.bm25,
                    k=settings.FINAL_K
                )

                if context:
                    # 2. Build and run LLM prompt
                    prompt = build_prompt(context, query)
                    answer = generate_answer(prompt)

                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                    # 3. Source citations in a clean expander
                    # Inside your Streamlit source loop
                    with st.expander("Show Evidence"):
                        for i, chunk in enumerate(context):
                            page = chunk['metadata'].get('page', 'N/A')
                            # Convert 0.9234 to "92.3%"
                            score_pct = f"{chunk.get('score', 0) * 100:.1f}%"

                            st.write(f"**Source {i + 1} (Page {page})** | Confidence: {score_pct}")
                            st.caption(chunk['text'])
                            st.divider()
                else:
                    msg = "I'm sorry, I couldn't find any information in the document related to that."
                    st.warning(msg)
                    st.session_state.messages.append({"role": "assistant", "content": msg})

else:
    # Welcome screen when no file is uploaded
    st.info("👋 Welcome! Please upload a PDF in the sidebar to start asking questions.")