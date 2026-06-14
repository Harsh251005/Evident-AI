from pdf_loader import load_pdf
from chunker import chunk_documents
from embedder import embed_texts

def ingestion_test(file_path: str) -> None:
    docs = load_pdf(file_path)
    chunks = chunk_documents(docs)
    texts = [chunk.page_content for chunk in chunks]
    embedded_chunks = embed_texts(texts)

ingestion_test(file_path=r"D:\Harsh\Code\Resume Projects\EvidentAI\data\temp_Personal AI Assistant Story.pdf")