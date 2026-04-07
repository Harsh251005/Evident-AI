from src.ingestion.embedder import embed_texts


class RagasEmbeddingWrapper:
    def embed_documents(self, texts):
        return embed_texts(texts)

    def embed_query(self, text):
        return embed_texts([text])[0]