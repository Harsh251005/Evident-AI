import os
import re
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Import your project settings
from config import settings

STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()

def tokenize(text: str):
    """Clean, tokenize, and stem text for BM25 processing."""
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return [STEMMER.stem(t) for t in tokens if t not in STOPWORDS]

def build_bm25_index(chunks):
    """Builds a BM25 index from document chunks."""
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]
    tokenized_corpus = [tokenize(t) for t in texts]

    bm25 = BM25Okapi(tokenized_corpus)
    # We return a dictionary to store texts and metadata alongside the index
    return {"bm25": bm25, "texts": texts, "metadata": metadatas}

def setup_bm25(collection_name, chunks=None):
    """
    Loads an existing BM25 index or builds a new one if chunks are provided.
    Uses settings.INDEX_DIR to ensure path consistency.
    """
    # 1. Use the directory defined in your settings.py
    index_path = os.path.join(settings.INDEX_DIR, f"{collection_name}_bm25.pkl")

    # 2. Try to load from disk
    if os.path.exists(index_path):
        print(f"[INFO] Loading existing BM25 index from: {index_path}")
        with open(index_path, "rb") as f:
            return pickle.load(f)

    # 3. If not found, build it (only if chunks were passed)
    if chunks is not None:
        print(f"[INFO] Building new BM25 index for {collection_name}...")
        index_data = build_bm25_index(chunks)

        # Ensure the index directory exists before saving
        os.makedirs(settings.INDEX_DIR, exist_ok=True)
        with open(index_path, "wb") as f:
            pickle.dump(index_data, f)
        print(f"[INFO] BM25 index saved successfully.")
        return index_data

    # 4. Critical Fail: No file and no data to make one
    raise ValueError(f"BM25 Index not found at {index_path} and no chunks provided to build one. "
                     "Please run ingestion via main.py first.")

def bm25_search(index_data, query, k=10):
    """
    Searches the BM25 index and returns formatted results with scores.
    """
    bm25 = index_data["bm25"]
    texts = index_data["texts"]
    metadatas = index_data["metadata"]

    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    # Sort indices based on score in descending order
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:k]

    return [
        {
            "text": texts[i],
            "metadata": metadatas[i],
            "score": float(scores[i])
        }
        for i in top_indices
    ]