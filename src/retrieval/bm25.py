import os
import pickle
import re
from rank_bm25 import BM25Okapi


def tokenize(text: str):
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    def __init__(self, documents):
        self.texts = [doc.page_content for doc in documents]
        self.metadata = [doc.metadata for doc in documents]

        tokenized_corpus = [tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query, k=10):
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]

        return [
            {
                "text": self.texts[i],
                "metadata": self.metadata[i],
                "score": scores[i]
            }
            for i in top_indices
        ]

    def save(self, file_path: str):
        """Saves the BM25 index to a file, ensuring the directory exists."""
        try:
            # 1. Ensure the directory exists (e.g., 'data/indices/')
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

            # 2. Save the file
            with open(file_path, "wb") as f:
                pickle.dump(self, f)
            print(f"[INFO] BM25 index saved to {file_path}")

        except Exception as e:
            # Catching generic Exception is better for saving
            # to catch Permission errors or Disk Full errors too
            print(f"[ERROR] Could not save BM25 index: {e}")

    @staticmethod
    def load(file_path: str):
        """Loads a BM25 index from a file."""
        if not os.path.exists(file_path):
            return None
        with open(file_path, "rb") as f:
            return pickle.load(f)