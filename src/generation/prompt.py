SYSTEM_PROMPT = """
You are a helpful AI assistant.

Answer the user's question using ONLY the provided context.

Rules:
1. If the answer is present in the context, answer accurately.
2. If the answer is not present in the context, say:
   "I could not find this information in the provided document."
3. Do not make up facts.
4. Keep answers concise but complete.
5. Cite page numbers when available.
"""


def build_context(context_docs: list) -> str:
    """
    Convert retrieved documents into a formatted context block.
    """

    sections = []

    for doc in context_docs:
        page_no = doc.metadata.get("page_no", "Unknown")

        sections.append(
            f"""
Page: {page_no}

{doc.page_content}
"""
        )

    return "\n\n---\n\n".join(sections)


def build_user_prompt(
    query: str,
    context_docs: list,
) -> str:
    """
    Build the final user prompt.
    """

    context = build_context(context_docs)

    return f"""
Context:
{context}

Question:
{query}

Answer:
"""