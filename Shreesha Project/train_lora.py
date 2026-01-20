from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

MODEL_NAME = "google/gemma-2b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    device_map="auto"
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Example dataset
data = [{"instruction":"What is AI?","response":"Artificial Intelligence is..."}]
from datasets import Dataset
train_dataset = Dataset.from_list(data)

def preprocess(example):
    text = f"Q: {example['instruction']}\nA: {example['response']}"
    tokens = tokenizer(text, truncation=True, padding="max_length", max_length=256)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

train_dataset = train_dataset.map(preprocess, remove_columns=["instruction","response"])

training_args = TrainingArguments(
    per_device_train_batch_size=2,
    max_steps=100,
    output_dir="./gemma-lora"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()
