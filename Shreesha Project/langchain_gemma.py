from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline

# 🔹 Choose base or fine-tuned model
MODEL_NAME = "google/gemma-2b"            # base model
# MODEL_NAME = "Shreesha012/gemma-lora"   # use this if you uploaded your LoRA fine-tuned model

# Load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")

# Wrap in Hugging Face pipeline
gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    temperature=0.7,
    top_p=0.9,
)

# Use in LangChain
llm = HuggingFacePipeline(pipeline=gen_pipeline)

# 🔹 Take input from the user
user_prompt = input("Enter your prompt: ")

# Generate response
response = llm.invoke(user_prompt)
print("\n=== Model Output ===\n")
print(response)
