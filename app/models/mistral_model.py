import os

from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv()

client = Mistral(
    api_key=os.getenv("MISTRAL_API_KEY")
)


def generate_response(prompt):
    """
    Generate response using Mistral AI.
    """

    response = client.chat.complete(

        model="mistral-small-latest",

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