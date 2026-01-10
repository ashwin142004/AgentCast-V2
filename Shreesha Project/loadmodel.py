from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from huggingface_hub import login
import os

login(token=os.getenv("hg_token"))

MODEL_NAME = "google/gemma-2b"  # use Gemma if you have access

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    device_map="auto"
)

# Apply LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Tiny dataset
# data = {
#     "train": [
#         {"instruction": "What is AI?", "response": "AI is the simulation of human intelligence by machines."},
#         {"instruction": "Who is Alan Turing?", "response": "Alan Turing was a mathematician and computer scientist, often called the father of AI."},
#     ]
# }

data = {
    "train": [
        {"text": "Artificial intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems. Applications of AI include expert systems, natural language processing, speech recognition, and machine vision."},
        {"text": "Alan Turing was a British mathematician and computer scientist. He is widely considered the father of theoretical computer science and artificial intelligence. He developed the Turing machine and played a key role in breaking German ciphers during World War II."},
    ]
}

train_dataset = Dataset.from_list(data["train"])

def preprocess(example):
    text = example["text"] + tokenizer.eos_token
    tokenized = tokenizer(text, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

train_dataset = Dataset.from_list(data["train"])
train_dataset = train_dataset.map(preprocess, remove_columns=["text"])

# Training args
args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=50,
    max_steps=200,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    output_dir="./lora-model",
    save_total_limit=2,
    save_steps=100,
    push_to_hub=True,
    hub_model_id="Shreesha012/gemma-lora"  # or your model name
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

trainer.train()
trainer.push_to_hub()
