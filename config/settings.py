import os
import yaml
from dotenv import load_dotenv

# 1. LOAD ENVIRONMENT VARIABLES FIRST
load_dotenv()

# --- DIRECTORIES (Dynamic Pathing) ---
# This makes your project work on ANY computer, not just your D: drive
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(DATA_DIR, "indices")
PROMPTS_PATH = os.path.join(BASE_DIR, "config", "prompts.yaml")

# --- SAMPLE DATA PATHS (Relative to project root) ---
STORY_PDF_PATH = os.path.join(DATA_DIR, "sample_pdf", "Story.pdf")
CLAUDE_CONSTITUTION_PDF_PATH = os.path.join(DATA_DIR, "sample_pdf", "claudes-constitution_webPDF_26-02.02a.pdf")

# --- MODEL SETTINGS ---
OPENAI_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
RERANKER_MODEL = "BAAI/bge-reranker-base"
VECTOR_SIZE = 1536

# --- CHUNKING SETTINGS ---
CHUNK_SIZE = 700
CHUNK_OVERLAP = 150

# --- RETRIEVAL SETTINGS ---
INITIAL_K = 10        # Number of docs to fetch before reranking
FINAL_K = 3          # Number of docs to send to the LLM after reranking
SCORE_THRESHOLD = 0.50 # Minimum confidence to consider a chunk
TOP_K = 3

# --- INFRASTRUCTURE ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

# --- LANGSMITH MONITORING ---
# We pull these from .env for security. Defaulting tracing to 'false' if key is missing.
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "EvidentAI-V1")

# --- PROMPTS LOADER ---
def load_prompts():
    if not os.path.exists(PROMPTS_PATH):
        raise FileNotFoundError(f"Prompts file not found at {PROMPTS_PATH}")
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

PROMPTS = load_prompts()