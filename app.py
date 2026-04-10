import streamlit as st
import nltk

nltk.download('stopwords')

from src.pipeline.evident_rag import EvidentAIRAG
from src.retrieval.dynamic_ingest import process_user_upload


st.set_page_config(page_title="EvidentAI", page_icon="🔍")

st.title("🔍 EvidentAI | Personal Document Assistant")

# 1. Force the user to upload before chatting
with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload your PDF to begin", type="pdf")

    if uploaded_file:
        with st.spinner("Analyzing and indexing your document..."):
            # This must return the unique hash-based name
            target_collection = process_user_upload(uploaded_file)
            st.success(f"Indexed: {uploaded_file.name}")
    else:
        st.info("Please upload a PDF to start chatting.")
        st.stop()  # Stops the rest of the app from running


# 2. Initialize RAG only for the uploaded file
@st.cache_resource
def get_rag(collection):
    # No default value here - it must take what the uploader gives it
    return EvidentAIRAG(collection_name=collection)


rag_engine = get_rag(target_collection)

# 3. Chat Interface (Standard logic)
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about your document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # This will now ONLY search the user_collection
        response = rag_engine.chain.invoke(prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})