import os
import yaml
from dotenv import load_dotenv

load_dotenv() # Load your .env file

CLAUDE_CONSTITUTION_PDF_PATH=r"D:\Harsh\Code\Resume Projects\EvidentAI\data\sample_pdf\claudes-constitution_webPDF_26-02.02a.pdf"
STORY_PDF_PATH=r"D:\Harsh\Code\Resume Projects\EvidentAI\data\sample_pdf\Story.pdf"

# --- MODEL SETTINGS ---
OPENAI_MODEL="gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

VECTOR_SIZE = 1536

# --- CHUNKING SETTINGS ---
CHUNK_SIZE = 700
CHUNK_OVERLAP = 150

# --- RETRIEVAL SETTINGS ---
INITIAL_K = 20
FINAL_K = 5
SCORE_THRESHOLD = 0.35
TOP_K = 5

# --- INFRASTRUCTURE ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# --- DIRECTORIES ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(DATA_DIR, "indices")
PROMPTS_PATH = os.path.join(BASE_DIR, "config", "prompts.yaml")

# --- PROMPTS ---
def load_prompts():
    with open(PROMPTS_PATH, "r") as f:
        return yaml.safe_load(f)

PROMPTS = load_prompts()