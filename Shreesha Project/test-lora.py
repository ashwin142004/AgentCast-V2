from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b",
    load_in_4bit=True,
    device_map="auto"
)

model = PeftModel.from_pretrained(
    base,
    "Shreesha012/gemma-podcast-lora"
)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")

prompt = """You are a podcast guest.
Question: What initially attracted you to working in AI?
Answer:"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

out = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.8,
    top_p=0.9,
    do_sample=True
)

print(tokenizer.decode(out[0], skip_special_tokens=True))
