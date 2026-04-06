import re
from rank_bm25 import BM25Okapi


def tokenize(text: str):
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    def __init__(self, texts, metadata):
        self.texts = texts
        self.metadata = metadata

        tokenized_corpus = [tokenize(t) for t in texts]
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