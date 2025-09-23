# modules/config.py
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load .env into environment
load_dotenv()

# Get API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not set in .env or environment")

# Shared OpenAI client
client = OpenAI(api_key=api_key)
