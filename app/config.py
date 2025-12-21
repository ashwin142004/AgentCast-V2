import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

genai.configure(api_key=GOOGLE_API_KEY)

# Model Definitions
# Using Gemini Flash Latest
MODEL_NAME = "gemini-flash-latest"

# Generation Config
GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
}

from langchain_google_genai import ChatGoogleGenerativeAI

# Shared LLM instance for agents
llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=0.7,
    google_api_key=GOOGLE_API_KEY
)

# JSON-mode LLM for Judge
judge_llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=0.0,
    google_api_key=GOOGLE_API_KEY
)
