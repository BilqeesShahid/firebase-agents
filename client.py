# client.py
from agents import AsyncOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Load Gemini API key
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Create reusable client (Google’s Gemini endpoint)
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
