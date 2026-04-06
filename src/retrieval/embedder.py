from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

def embed_texts(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [e.embedding for e in response.data]