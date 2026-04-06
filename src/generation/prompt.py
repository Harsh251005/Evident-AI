def build_prompt(context, query):
    context_text = "\n\n".join([c["text"] for c in context])

    return f"""
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context_text}

Question:
{query}

Answer:
"""