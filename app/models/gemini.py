import os

from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Read API Key
API_KEY = os.getenv("GEMINI_API_KEY")

# Create Gemini Client
client = genai.Client(api_key=API_KEY)


def generate_response(prompt):
    """
    Send a prompt to Gemini and return the response.
    """

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
    )

    return response.text


if __name__ == "__main__":

    prompt = """
    Solve this maths problem. 
    
    If a fair coin is tossed once, what is the probability of getting Heads?

    Explain your reasoning.
"""

    response = generate_response(prompt)

    print("\nGemini Response:\n")
    print(response)