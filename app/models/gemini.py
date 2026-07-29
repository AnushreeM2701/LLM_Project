import os
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Read API Key
API_KEY = os.getenv("GEMINI_API_KEY")

# Create Gemini Client
client = genai.Client(api_key=API_KEY)


def generate_response(prompt: str) -> str:
    """
    Generate a response using Gemini.
    """

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
    )

    if response.text:
        return response.text.strip()

    return ""


if __name__ == "__main__":

    question = """
If a fair coin is tossed once, what is the probability of obtaining heads?
"""

    print(generate_response(question))