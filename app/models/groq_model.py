import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def generate_response(prompt):
    """
    Generate response using Groq.
    """

    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

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