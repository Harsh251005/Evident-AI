import time

from openai import OpenAI, APIError, APIConnectionError
from langchain_core.documents import Document
from langsmith import traceable

from config.settings import settings
from src.generation.prompt import (
    SYSTEM_PROMPT,
    build_user_prompt,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

client = OpenAI(timeout=60.0)

_MAX_ATTEMPTS = 3


@traceable(
    name="generation",
    run_type="llm",
    metadata={
        "ls_provider": settings.PROVIDER,
        "ls_model": settings.OPENAI_MODEL,
    }
)
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

    last_error = None

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            response = client.responses.create(
                model=settings.OPENAI_MODEL,
                instructions=SYSTEM_PROMPT,
                input=user_prompt,
            )

            answer = response.output_text.strip()

            logger.info("Answer generated successfully")

            return answer

        except (APIConnectionError, APIError) as e:
            last_error = e
            if attempt == _MAX_ATTEMPTS:
                break
            logger.warning(
                f"Generation failed (attempt {attempt}/{_MAX_ATTEMPTS}): {e} — retrying"
            )
            time.sleep(2 * attempt)

        except Exception as e:
            logger.exception("Generation failed")
            raise RuntimeError(
                "Failed to generate answer"
            ) from e

    logger.exception("Generation failed after retries")
    raise RuntimeError(
        "Failed to generate answer"
    ) from last_error