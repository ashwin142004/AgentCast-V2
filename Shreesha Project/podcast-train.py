from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
print("DEBUG: Transformers imported")
from peft import LoraConfig, get_peft_model
print("DEBUG: PEFT imported")
from datasets import load_dataset, concatenate_datasets
print("DEBUG: Datasets imported")
from huggingface_hub import login
print("DEBUG: HF Hub imported")
import torch
print("DEBUG: Torch imported")
import os
print("DEBUG: OS imported")


# ✅ Login to Hugging Face
login(token=os.getenv("hg_token"))  # your token

MODEL_NAME = "google/gemma-2b"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ✅ CORRECTED: Use BitsAndBytesConfig for 4-bit loading
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto"
)

# ✅ Apply LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# ✅ CORRECTED: Robust dataset loading
# We start with an empty list and only append datasets that load successfully.
loaded_datasets = []

print("Starting dataset loading...")

# --- Load Dataset 1 ---
# try:
#     # Note: peoples_speech is very large. Slicing 1% is still huge.
#     # Using 'train[:1%]' for demonstration. You may want a smaller slice like 'train[:10000]'
#     ds1 = load_dataset("MLCommons/peoples_speech", split="train[:1%]")
#     loaded_datasets.append(ds1)
#     print("Successfully loaded MLCommons/peoples_speech")
# except Exception as e:
#     print(f"Failed to load MLCommons/peoples_speech: {e}")

# --- Load Dataset 2 ---
try:
    print("Attempting to load Whispering-GPT/lex-fridman-podcast-transcript-audio...")
    # Added verification_mode="no_checks" to bypass metadata mismatch errors
    ds2 = load_dataset("Whispering-GPT/lex-fridman-podcast-transcript-audio", split="train[:100]", verification_mode="no_checks")
    loaded_datasets.append(ds2)
    print("Successfully loaded Whispering-GPT/lex-fridman-podcast-transcript-audio")
except Exception as e:
    print(f"Failed to load Whispering-GPT/lex-fridman-podcast-transcript-audio: {e}")

# --- Load Gated Dataset 3 ---
# try:
#     # ⚠️ ACTION REQUIRED: You must request access at the URL below
#     ds3 = load_dataset("blitt/SPoRC", split="train[:5%]")
#     loaded_datasets.append(ds3)
#     print("Successfully loaded blitt/SPoRC")
# except Exception as e:
#     print(f"Failed to load blitt/SPoRC: {e}")
#     print("NOTE: 'blitt/SPoRC' is a gated dataset. You must log in and request access at:")
#     print("   https://huggingface.co/datasets/blitt/SPoRC")


# ✅ Check if any datasets were loaded before continuing
if not loaded_datasets:
    print("No datasets were successfully loaded. Exiting.")
    exit()

print(f"\nCombining {len(loaded_datasets)} dataset(s)...")
combined = concatenate_datasets(loaded_datasets)

# ✅ Unify into a single text field
def unify_text(example):
    if "transcript" in example and example["transcript"]:
        text = example["transcript"]
    elif "text" in example and example["text"]:
        text = example["text"]
    elif "dialogue" in example and example["dialogue"]:
        text = example["dialogue"]
    else:
        # Fallback just in case, to avoid errors with NoneType
        text = str(example) 
    return {"text": text}

# Remove all original columns (including heavy audio) when creating the 'text' column
# This prevents PyArrow overflow errors due to large binary data
combined = combined.map(unify_text, remove_columns=combined.column_names)

# ✅ Tokenize
def preprocess(example):
    # Ensure text is not None before proceeding
    text_to_process = example["text"] if example["text"] else ""
    text = text_to_process + tokenizer.eos_token
    
    tokenized = tokenizer(text, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

train_dataset = combined.map(preprocess, remove_columns=["text"]) # Remove the intermediate 'text' column

# ✅ Training args
args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=50,
    max_steps=200,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    output_dir="./podcast-lora",
    save_total_limit=2,
    save_steps=100,
    push_to_hub=True,
    hub_model_id="Shreesha012/gemma-podcast-lora" # Make sure this repo exists on your HF account
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

print("Starting fine-tuning on podcast dataset...")
trainer.train()
trainer.push_to_hub()
print("Training complete and model pushed to Hugging Face!")
