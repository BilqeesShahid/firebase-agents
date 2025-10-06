# client.py
import os
import json
from agents import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# Use GEMINI API Key if you want key-based access (optional)
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Create reusable client for Gemini (OpenAI-compatible endpoint)
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
