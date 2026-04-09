import yaml
import os
from config import settings
from src.retrieval.hybrid import hybrid_search
from src.retrieval.bm25 import setup_bm25
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


class EvidentAIRAG:
    def __init__(self, collection_name):
        self.collection_name = collection_name

        # 1. Load the specific BM25 index for this user upload
        self.bm25_data = setup_bm25(self.collection_name)

        # 2. Initialize the LLM (gpt-4o-mini is best for speed/cost)
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # 3. Load the prompt from your YAML file directly
        prompt_path = "config/prompts.yaml"
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Missing prompt configuration at {prompt_path}")

        with open(prompt_path, "r") as f:
            prompts = yaml.safe_load(f)

        # 4. Define the prompt template the chain will use
        self.prompt_template = ChatPromptTemplate.from_template(
            prompts.get("qa_system_prompt", "{context}\n\nQuestion: {question}")
        )

    def retrieve_docs(self, query: str):
        try:
            results = hybrid_search(
                query=query,
                collection_name=self.collection_name,
                bm25_index=self.bm25_data,
                k=settings.INITIAL_K
            )
        except Exception as e:
            print(f"⚠️ Retrieval Error: {e}")
            return "The system is still indexing your document. Please wait a moment."

        if not results:
            return "No relevant context found in the uploaded document."

        # Build the formatted string for the LLM context window
        formatted_context = ""
        for i, doc in enumerate(results):
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            page_num = metadata.get("page", "Unknown")

            formatted_context += f"\n--- SOURCE {i + 1} (Page {page_num}) ---\n"
            formatted_context += f"{text}\n"

        return formatted_context

    @property
    def chain(self):
        # Now self.prompt_template and self.llm are defined, this will work!
        return (
                {"context": self.retrieve_docs, "question": RunnablePassthrough()}
                | self.prompt_template
                | self.llm
                | StrOutputParser()
        )