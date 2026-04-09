import re
import tiktoken


def calculate_citation_coverage(answer: str) -> float:
    """
    Calculates the percentage of sentences that contain a valid (Page X) citation.
    Supports multipage formats like (Page 1, Page 2).
    """
    if not answer or not isinstance(answer, str):
        return 0.0

    # 1. Split into sentences (handles . ! ? and trailing whitespace)
    # We filter out very short strings that aren't real sentences
    sentences = [s.strip() for s in re.split(r'[.!?]', answer) if len(s.strip()) > 10]

    if not sentences:
        return 1.0

    # 2. Robust Regex: Matches (Page 1), (Page 1, Page 2), or (Page 1, 2)
    # This is more forgiving than the previous version
    citation_pattern = r"\(Page \d+(?:,?\s*(?:Page\s*)?\d+)*\)"

    cited_count = 0
    for s in sentences:
        if re.search(citation_pattern, s, re.IGNORECASE):
            cited_count += 1

    return round(cited_count / len(sentences), 2)


def calculate_request_cost(input_text: str, output_text: str) -> float:
    """
    Calculates cost based on GPT-4o-mini production rates.
    Input: $0.15 / 1M tokens | Output: $0.60 / 1M tokens
    """
    try:
        enc = tiktoken.encoding_for_model("gpt-4o-mini")

        in_tokens = len(enc.encode(input_text))
        out_tokens = len(enc.encode(output_text))

        # GPT-4o-mini Pricing (Approximate)
        cost = (in_tokens * (0.15 / 1_000_000)) + (out_tokens * (0.60 / 1_000_000))

        return round(cost, 6)
    except Exception:
        return 0.0