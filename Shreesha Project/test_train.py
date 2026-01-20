from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from huggingface_hub import login
import torch
import os
from dotenv import load_dotenv

load_dotenv()

print("--- DRY RUN TRAINING SCRIPT STARTING ---")

# 1. Login
login(token=os.getenv("hg_token"))

MODEL_NAME = "google/gemma-2b"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Set chat template globally for base model
tokenizer.chat_template = (
    "{{ bos_token }}"
    "{% set loop_messages = messages %}"
    "{% if messages[0]['role'] == 'system' %}"
    "{{ '<start_of_turn>system\\n' + messages[0]['content'] | trim + '<end_of_turn>\\n' }}"
    "{% set loop_messages = messages[1:] %}"
    "{% endif %}"
    "{% for message in loop_messages %}"
    "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
    "{% endif %}"
    "{% if message['role'] == 'user' %}"
    "{{ '<start_of_turn>user\\n' + message['content'] | trim + '<end_of_turn>\\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<start_of_turn>model\\n' + message['content'] | trim + '<end_of_turn>\\n' }}"
    "{% else %}"
    "{{ raise_exception('Only user and assistant roles are allowed after initial system message') }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<start_of_turn>model\\n' }}"
    "{% endif %}"
)

# 2. Config & Model Load
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print(f"Loading model: {MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

# 3. LoRA Config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 4. Load Dataset (Local JSONL)
DATA_FILE = "lex_guest_qa.jsonl"
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Missing dataset file: {DATA_FILE}. Run build_dataset.py first.")

print(f"Loading local dataset: {DATA_FILE}")
dataset = load_dataset("json", data_files=DATA_FILE, split="train")
# Use small subset for dry run if huge
dataset = dataset.select(range(min(len(dataset), 50)))
print(f"Dataset loaded. Dry run size: {len(dataset)} samples.")

# 5. Format & Tokenize
def format_and_tokenize(example):
    # Template already set globally
    pass

    text = tokenizer.apply_chat_template(
        example["messages"], 
        tokenize=False, 
        add_generation_prompt=False
    )
    tokenized = tokenizer(
        text, 
        truncation=True, 
        padding="max_length", 
        max_length=512
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

print("Tokenizing dataset...")
train_dataset = dataset.map(format_and_tokenize, remove_columns=["messages"])

# 6. Training Arguments (DRY RUN)
output_dir = "./podcast-lora-dryrun"

args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    warmup_steps=0,
    max_steps=5,              # DRY RUN: 5 steps
    learning_rate=1e-4,
    fp16=True,
    logging_steps=1,
    output_dir=output_dir,
    save_total_limit=1,
    save_steps=5,
    push_to_hub=False,        # DISABLE PUSH
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# 7. Train
print("Starting training (DRY RUN)...")
trainer.train()

print("✅ Dry run complete!")
