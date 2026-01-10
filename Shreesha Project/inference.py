from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch

# Path to your fine-tuned LoRA model
BASE_MODEL = "google/gemma-2b"
LORA_MODEL = "./lora-model"  # folder where trainer saved LoRA weights

# Load base model in 4-bit, FP16 for speed
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)

# Apply LoRA weights
model = PeftModel.from_pretrained(model, LORA_MODEL, device_map="auto")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Interactive loop
print("✅ AI Q&A (type 'exit' to quit)\n")

while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit", "q"]:
        print("👋 Exiting...")
        break

    # Prepare input
    prompt = f"### Question:\n{query}\n### Answer:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,       # randomness for varied answers
        temperature=0.7,      # creativity level
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode and clean output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response[len(prompt):].strip()
    print(f"AI: {answer}\n")