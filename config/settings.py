import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    PROVIDER: str = "openai"
    # Generation only. The RAGAS + LLM-as-judge eval pipeline (run_eval.py)
    # is pinned separately to gpt-4.1-mini — ragas==0.4.3's agenerate() sends
    # the legacy `max_tokens` param, which gpt-5.6-luna rejects (requires
    # `max_completion_tokens`), a library-level incompatibility with no fix
    # available upstream yet. Confirmed via a direct RAGAS call, not assumed.
    OPENAI_MODEL: str = "gpt-5.6-luna"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY")
    QDRANT_URL: str = os.getenv("QDRANT_URL")
    QDRANT_COLLECTION_NAME: str = "claudes-constitution_webpdf_26-02.02a_09559b3b"

    # Reranking — cross-encoder re-scoring of hybrid retrieval candidates
    RERANKER_MODEL: str = "ms-marco-MiniLM-L-12-v2"  # via FlashRank, CPU-optimized
    INITIAL_K: int = 10   # candidates fetched before reranking
    FINAL_K: int = 4      # "Golden 4" — chunks sent to the LLM after reranking

    LANGSMITH_TRACING: str = os.getenv("LANGSMITH_TRACING")
    LANGSMITH_ENDPOINT: str = os.getenv("LANGSMITH_ENDPOINT")
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT")


settings = Settings()


_env_map = {
    "OPENAI_API_KEY":     settings.OPENAI_API_KEY,
    "QDRANT_API_KEY":     settings.QDRANT_API_KEY,
    "QDRANT_URL":         settings.QDRANT_URL,
    "LANGSMITH_TRACING":  settings.LANGSMITH_TRACING,
    "LANGSMITH_ENDPOINT": settings.LANGSMITH_ENDPOINT,
    "LANGSMITH_API_KEY":  settings.LANGSMITH_API_KEY,
    "LANGSMITH_PROJECT":  settings.LANGSMITH_PROJECT,
}

for _key, _value in _env_map.items():
    if _value is not None:
        os.environ[_key] = _value