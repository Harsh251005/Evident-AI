import fitz  # PyMuPDF

def load_pdf(file_path: str):
    docs = []
    pdf = fitz.open(file_path)

    for page_num, page in enumerate(pdf):
        text = page.get_text()
        docs.append({
            "text": text,
            "page": page_num + 1
        })

    return docs