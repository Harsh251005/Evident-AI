from config import settings
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

def embed_texts(texts):
    response = client.embeddings.create(
        model=settings.EMBEDDING_MODEL,
        input=texts
    )
    return [e.embedding for e in response.data]