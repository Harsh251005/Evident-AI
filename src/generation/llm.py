from config import settings
from openai import OpenAI

client = OpenAI(api_key=settings.OPENAI_API_KEY)

def generate_answer(prompt):
    response = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content