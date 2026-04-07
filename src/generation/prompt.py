from langchain_core.prompts import PromptTemplate
from config.settings import PROMPTS


def build_prompt(context_docs: list, query: str):
    # Instead of just joining text, we label each piece
    context_parts = []

    for doc in context_docs:
        page_num = doc["metadata"].get("page", "Unknown")
        content = doc["text"]

        # Prepend the page number to the text
        context_parts.append(f"[Source: Page {page_num}]\n{content}")

    # Join with double newlines for clarity
    context_text = "\n\n---\n\n".join(context_parts)

    # Load the raw string from your YAML
    template_str = PROMPTS["qa_system_prompt"]

    # Create a LangChain PromptTemplate
    prompt_template = PromptTemplate.from_template(template_str)

    # Format the prompt safely
    final_prompt = prompt_template.format(
        context=context_text,
        question=query
    )

    return final_prompt