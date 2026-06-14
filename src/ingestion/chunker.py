from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.logger import get_logger

logger = get_logger(__name__)


TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=200,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ],
    length_function=len,
    add_start_index=True,
)


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into overlapping chunks for retrieval.
    """

    chunks = TEXT_SPLITTER.split_documents(documents)

    logger.info(
        f"Chunked {len(documents)} documents into {len(chunks)} chunks"
    )

    return chunks
