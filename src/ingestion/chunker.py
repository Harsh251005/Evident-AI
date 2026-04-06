def chunk_text(documents, chunk_size=500, overlap=100):
    chunks = []

    for doc in documents:
        text = doc["text"]
        page = doc["page"]

        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            chunks.append({
                "text": chunk,
                "page": page
            })

            start += chunk_size - overlap

    return chunks