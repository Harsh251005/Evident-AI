from src.pipeline.evident_rag import EvidentAIRAG
import logging

# Setup basic logging to see what's happening
logging.basicConfig(level=logging.INFO)


def test_e2e_pipeline():
    print("🚀 Initializing EvidentAI Production Pipeline...")
    rag = EvidentAIRAG()

    question = "Who are the authors of the claude's constitution?"  # Change to a real question from your PDF

    print(f"\n🔍 Testing Question: {question}")
    print("-" * 30)

    # This triggers Hybrid Retrieval -> Reranking -> LLM
    response = rag.chain.invoke(question)

    print(f"\n🤖 AI Response:\n{response}")
    print("-" * 30)

    # VALIDATION CHECKLIST
    if "(Page" in response:
        print("✅ CITATION CHECK: PASSED")
    else:
        print("❌ CITATION CHECK: FAILED (No page numbers found)")

    if len(response) > 50:
        print("✅ RESPONSE QUALITY: PASSED")
    else:
        print("❌ RESPONSE QUALITY: FAILED (Response too short)")


if __name__ == "__main__":
    test_e2e_pipeline()