import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import CrossEntropyLoss

# 🔹 Load the model you want to evaluate
MODEL_NAME = "google/gemma-2b"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")

# 🔹 Prepare your test dataset (a list of strings from your podcast data)
# This data MUST NOT have been used in training.
test_data = [
    "The guest explained that quantum computing isn't about speed, but about solving new types of problems.",
    "Welcome back to the show, today our expert will demystify black holes.",
    # ... add more samples from your held-out test set
]

total_perplexity = 0
num_samples = 0

for text in test_data:
    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    
    with torch.no_grad():
        # Get the model's logits
        outputs = model(**inputs, labels=input_ids)
        loss = outputs.loss
        
        # Perplexity is the exponential of the loss
        perplexity = torch.exp(loss)
        
        total_perplexity += perplexity.item()
        num_samples += 1

# Calculate the average perplexity across all samples
average_perplexity = total_perplexity / num_samples
print(f"Average Perplexity: {average_perplexity:.2f}")
