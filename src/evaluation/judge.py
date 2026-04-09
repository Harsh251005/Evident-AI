from src.generation.llm import generate_answer # Or use a direct OpenAI call
from config.settings import PROMPTS

JUDGE_PROMPT = PROMPTS['judge_prompt']

import re


def grade_with_llm(ground_truth, generated_answer):
    prompt = JUDGE_PROMPT.format(
        ground_truth=ground_truth,
        generated_answer=generated_answer
    )
    response = generate_answer(prompt)

    # Use Regex to find the numeric score (1-5)
    match = re.search(r"Score:\s*([1-5])", response)
    score = int(match.group(1)) if match else None

    return score, response