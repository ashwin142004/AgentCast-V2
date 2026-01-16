import re
import json
import os
from datasets import load_dataset, Dataset

# -----------------------------
# CONFIG
# -----------------------------
# We load the SPoRC speaker turns file directly.
# This URL is from the repository structure we inspected.
SPORC_URL = "https://huggingface.co/datasets/blitt/SPoRC/resolve/main/speakerTurnData.jsonl.gz"
OUTPUT_FILE = "lex_guest_qa.jsonl"
MIN_SAMPLES = 5000  # Target number of samples

SYSTEM_PROMPT = (
    "You are a podcast guest. "
    "You answer questions naturally, thoughtfully, "
    "and in spoken English."
)

def build_dataset():
    print(f"Loading SPoRC Speaker Turns (User must be logged in via 'huggingface-cli login' or login.py)...")
    
    # Load streaming to handle size
    ds = load_dataset("json", data_files=SPORC_URL, split="train", streaming=True)

    all_samples = []
    
    current_episode_url = None
    last_turn_role = None 
    last_turn_text = None
    
    print("Extracting Host-Guest pairs from SPoRC...")
    
    # Initialize file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        pass
    
    count = 0
    collected = 0
    
    for i, entry in enumerate(ds):
        try:
            mp3_url = entry.get("mp3url")
            role = entry.get("inferredSpeakerRole") # 'host' or 'guest'
            text = entry.get("turnText", "").strip()
            
            # Skip if critical info missing
            if not mp3_url or not role or not text:
                continue
                
            # If episode changed, reset context
            if mp3_url != current_episode_url:
                current_episode_url = mp3_url
                last_turn_role = None
                last_turn_text = None
            
            # Logic: If current is 'guest' and previous was 'host', we have a Pair!
            # Host (User) -> Guest (Assistant)
            
            if role == "guest" and last_turn_role == "host" and last_turn_text:
                q_text = last_turn_text
                a_text = text
                
                # Filter short/garbage
                if len(q_text) > 10 and len(a_text) > 20: 
                     sample = {
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": q_text},
                            {"role": "assistant", "content": a_text}
                        ]
                    }
                     # Write immediately
                     with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                         f.write(json.dumps(sample) + "\n")
                     
                     collected += 1
            
            # Update state for next turn
            last_turn_role = role
            # If guest speaks multiple times, we might skip or append. 
            # For strict pairs, we just store current text.
            last_turn_text = text
            
            count += 1
            if count % 1000 == 0:
                print(f"Scanned {count} turns. Collected {collected} samples.")
                
            if collected >= 2000:
                print(f"Reached 2000 samples. Stopping.")
                break
                
        except Exception as e:
            print(f"Error processing row {i}: {e}")
            continue

    print(f"Done. Collected {collected} samples.")

if __name__ == "__main__":
    build_dataset()
