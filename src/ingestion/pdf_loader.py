import fitz
from langchain_core.documents import Document

def load_pdf(file_path: str):
    docs = []

    with fitz.open(file_path) as pdf:
        total_pages = len(pdf)

        for page_num, page in enumerate(pdf):
            text = page.get_text("text")

            # Wrap the data in a Document object instead of a dictionary
            docs.append(Document(
                page_content=text,
                metadata={
                    "source": file_path,
                    "page_no": page_num + 1,
                    "total_pages": total_pages
                }
            ))

    return docs