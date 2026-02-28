import os
import torch
from dotenv import load_dotenv
from google import genai
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

# Import Google integration for the Judge
from langchain_google_genai import ChatGoogleGenerativeAI

# Import Local Hugging Face integration
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

load_dotenv()

# --- 1. Setup API Keys ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")
if not HF_TOKEN:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in .env file.")

client = genai.Client(api_key=GOOGLE_API_KEY)

# --- 2. Model Definitions ---

# JUDGE Model -> Gemini Flash
# Initialize primary (Gemini) and fallback (Groq) models
gemini = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0.0,
    google_api_key=GOOGLE_API_KEY  # Use os.getenv for security
)

groq = ChatGroq(
    model="llama-3.1-8b-instant",  # Or another fast model like "mixtral-8x7b-32768"
    temperature=0.0,
    groq_api_key=GROQ_API_KEY
)

# Create judge_llm with fallback chain (handles rate limits automatically)
judge_llm = gemini.with_fallbacks([groq])

# HOST Model -> Groq Mixtral
print("Initializing Groq Mixtral Host Model...")
host_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    groq_api_key=GROQ_API_KEY
)

# GUEST Model -> Local Gemma 2B + LoRA wrapped in ChatHuggingFace
print("Initializing Local Gemma Guest Model...")

base_model_id = "google/gemma-2b-it"
adapter_id = "ashwin1414/gemma-podcast-lora"

# A. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=HF_TOKEN)

# B. Load Base Model (4-bit)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    load_in_4bit=True,
    token=HF_TOKEN
)

# C. Attach Adapter
model = PeftModel.from_pretrained(base_model, adapter_id, token=HF_TOKEN)

# D. Create the Pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.7,
    repetition_penalty=1.1,
    return_full_text=False
)

# E. Wrap in HuggingFacePipeline
hf_pipeline = HuggingFacePipeline(pipeline=pipe)

# F. Wrap in ChatHuggingFace -> This ensures the output has .content!
llm = ChatHuggingFace(llm=hf_pipeline)