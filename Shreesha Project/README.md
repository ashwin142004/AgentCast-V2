# AgentCast-V2 (Shreesha Project)

This project contains scripts for fine-tuning a Large Language Model (LLM) on podcast datasets using LoRA (Low-Rank Adaptation).

## Prerequisites

Ensure you have Python installed. It is recommended to use a virtual environment.

## Installation

1.  **Install Dependencies**
    
    This project relies on several libraries including `transformers`, `peft`, `datasets`, and `torch`.
    
    ```bash
    pip install -r requirements.txt
    pip install transformers peft torch
    ```
    
    *Note: If you run into issues with `bitsandbytes` on Windows, ensuring you have the correct CUDA version installed is important.*

2.  **Hugging Face Login**
    
    The script requires a Hugging Face token. You can either log in via CLI or ensure the token in `podcast-train.py` is valid.
    
    ```bash
    huggingface-cli login
    ```

## Usage

The main training script is `podcast-train.py`. This script loads the Gemma-2b model, applies LoRA, loads the podcast dataset, and fine-tunes the model.

To start the training process, run:

```bash
python podcast-train.py
```

## Output

The fine-tuned model adapters will be saved in the `podcast-lora/` directory.
