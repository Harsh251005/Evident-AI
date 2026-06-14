from pathlib import Path
import pymupdf

from langchain_core.documents import Document

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_pdf(file_path: str) -> list[Document]:
    """
    Load a PDF and return one Document per page.

    Metadata:
        - source
        - page_no
        - total_pages
    """

    pdf_path = Path(file_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    documents: list[Document] = []

    try:
        with pymupdf.open(pdf_path) as pdf:
            total_pages = len(pdf)

            for page_num, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()

                if not text:
                    continue

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": pdf_path.name,
                            "page_no": page_num,
                            "total_pages": total_pages,
                        },
                    )
                )

        logger.info(
            f"Loaded PDF '{pdf_path.name}' "
            f"({len(documents)}/{total_pages} non-empty pages)"
        )

        return documents

    except Exception as e:
        logger.exception(f"Failed to load PDF: {pdf_path}")
        raise RuntimeError(f"Failed to process PDF: {pdf_path}") from e
