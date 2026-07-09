import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)


def generate_response(prompt):
    """
    Generate response using GPT-OSS 120B
    through OpenRouter.
    """

    response = client.chat.completions.create(

        model="openai/gpt-oss-120b:free",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0

    )

    return response.choices[0].message.content


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    question = """
If a fair coin is tossed once,
what is the probability of getting Heads?
"""

    print(generate_response(question))