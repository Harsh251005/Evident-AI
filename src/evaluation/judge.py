from src.generation.llm import generate_answer


def judge_answer(question, ground_truth, generated):
    prompt = f"""
You are an evaluator.

Question: {question}

Ground Truth Answer:
{ground_truth}

Generated Answer:
{generated}

Evaluate:
1. Correctness (0-1)
2. Completeness (0-1)

Return JSON:
{{
  "correctness": ...,
  "completeness": ...
}}
"""

    response = generate_answer(prompt)
    return response