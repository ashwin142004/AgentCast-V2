(Example: simple Q&A dataset)
data = {
    "train": [
        {"instruction": "What is AI?", "response": "AI is the simulation of human intelligence by machines."},
        {"instruction": "Who is Alan Turing?", "response": "Alan Turing was a mathematician and computer scientist, often called the father of AI."},
    ]
}

from datasets import Dataset
train_dataset = Dataset.from_list(data["train"])

def preprocess(example):
    text = f"### Question:\n{example['instruction']}\n### Answer:\n{example['response']}"
    tokenized = tokenizer(text, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

train_dataset = train_dataset.map(preprocess, remove_columns=["instruction","response"])
