import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# Load environment variables from the .env file in the root directory
load_dotenv()

# --- API KEY CONFIGURATION ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

# --- MODEL CONFIGURATION ---
# Central place to define the models used by the agents.
HOST_LLM = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
GUEST_LLM = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# --- STRUCTURED OUTPUT CLASS FOR DCS ---
# We define a Pydantic model for the structured data we want the Judge LLM to return.
# This is a much more reliable method than prompting for raw JSON.
class DCSAnalysis(BaseModel):
    """A schema for the DCS analysis output."""
    coherence_score: int = Field(..., description="An integer from 1 (completely incoherent) to 10 (perfectly coherent).")
    coherence_reason: str = Field(..., description="A brief, one-sentence explanation for the coherence score.")
    topic_drift: str = Field(..., description="A string, either 'on_topic', 'slight_drift', or 'off_topic'.")

# The "Judge" LLM for DCS analysis, now configured for reliable structured output.
# We attach our DCSAnalysis schema to it.
JUDGE_LLM = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0
).with_structured_output(DCSAnalysis)

