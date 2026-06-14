from openai import OpenAI
from langchain_core.documents import Document

from config.settings import settings
from src.generation.prompt import (
    SYSTEM_PROMPT,
    build_user_prompt,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

client = OpenAI()


def generate_answer(
    query: str,
    context_docs: list[Document],
) -> str:
    """
    Generate an answer from retrieved context.
    """

    if not context_docs:
        return (
            "I could not find any relevant information "
            "in the document."
        )

    user_prompt = build_user_prompt(
        query=query,
        context_docs=context_docs,
    )

    logger.info(
        f"Generating answer using "
        f"{len(context_docs)} retrieved chunks"
    )

    try:
        response = client.responses.create(
            model=settings.OPENAI_MODEL,
            instructions=SYSTEM_PROMPT,
            input=user_prompt,
        )

        answer = response.output_text.strip()

        logger.info("Answer generated successfully")

        return answer

    except Exception as e:
        logger.exception("Generation failed")
        raise RuntimeError(
            "Failed to generate answer"
        ) from e