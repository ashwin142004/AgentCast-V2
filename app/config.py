import os
from dotenv import load_dotenv
from google import genai

# Import Google integration for the Judge
from langchain_google_genai import ChatGoogleGenerativeAI

# Import Hugging Face integration for Host/Guest
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

# --- 1. Setup API Keys ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")
if not HF_TOKEN:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in .env file. Please add it to your .env.")

client = genai.Client(api_key=GOOGLE_API_KEY)

# --- 2. Model Definitions ---

# JUDGE Model -> Uses Google's Gemini Flash
# We use Gemini here because it is stable and cost-effective for JSON evaluation.
judge_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,
    google_api_key=GOOGLE_API_KEY
)

# HOST & GUEST Model -> Uses Gemma 2 27B IT via Hugging Face
# We use HuggingFaceEndpoint to connect to the model hosted on HF's Inference API.
hf_endpoint = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    task="text-generation",
    max_new_tokens=2048,
    do_sample=True,
    temperature=0.7,
    huggingfacehub_api_token=HF_TOKEN
)

# CRITICAL FIX: Explicitly pass 'model_id' to ChatHuggingFace.
# This prevents the 'StopIteration' error by manually telling the wrapper which tokenizer to use.
llm = ChatHuggingFace(
    llm=hf_endpoint,
    model_id="google/gemma-2-9b-it"
)