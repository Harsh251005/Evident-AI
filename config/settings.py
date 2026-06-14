import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    PROVIDER: str = "openai"
    OPENAI_MODEL: str = "gpt-4.1-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY")
    QDRANT_URL: str = os.getenv("QDRANT_URL")
    QDRANT_COLLECTION_NAME: str = "claudes-constitution_webpdf_26-02.02a_09559b3b"

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