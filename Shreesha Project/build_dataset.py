import re
from datasets import load_dataset, Dataset

# -----------------------------
# CONFIG
# -----------------------------
DATASET_NAME = "Whispering-GPT/lex-fridman-podcast-transcript-audio"
OUTPUT_PATH = "./lex_guest_qa_dataset"

SYSTEM_PROMPT = (
    "You are a podcast guest. "
    "You answer questions naturally, thoughtfully, "
    "and in spoken English."
)

# -----------------------------
# Q/A EXTRACTION LOGIC
# -----------------------------
def extract_qa_from_transcript(transcript):
    if not transcript:
        return []

    lines = [l.strip() for l in transcript.split("\n") if l.strip()]
    samples = []

    current_question = None

    for line in lines:
        lower = line.lower()

        # Detect Lex asking a question (very flexible)
        if "lex" in lower and "?" in line:
            # Remove timestamps and speaker name
            q = re.sub(r"\[.*?\]", "", line)
            q = re.sub(r".*lex[^:]*:\s*", "", q, flags=re.I)

            if len(q) > 20:
                current_question = q.strip()
            continue

        # Detect an answer AFTER a question
        if current_question:
            # Skip Lex continuing to talk
            if "lex" in lower:
                continue

            # Clean answer line
            a = re.sub(r"\[.*?\]", "", line)
            a = re.sub(r".*?:\s*", "", a)

            if len(a) < 50:
                continue

            samples.append({
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a podcast guest. "
                            "You answer questions naturally, thoughtfully, "
                            "and in spoken English."
                        )
                    },
                    {"role": "user", "content": current_question},
                    {"role": "assistant", "content": a.strip()}
                ]
            })

            current_question = None

    return samples



# -----------------------------
# MAIN
# -----------------------------
def main():
    print("Loading Lex Fridman dataset...")

    ds = load_dataset(
        DATASET_NAME,
        split="train",
        verification_mode="no_checks"
    )

    # 🔥 CRITICAL FIX: REMOVE AUDIO COLUMN
    # We remove 'audio' specifically.
    if "audio" in ds.column_names:
        ds = ds.remove_columns("audio")

    print(f"Dataset columns after cleanup: {ds.column_names}")

    all_samples = []
    count = 0

    for i, ex in enumerate(ds):
        # Use 'text' instead of 'transcript'
        transcript = ex.get("text", "")
        
        # DEBUG: Print first transcript to verify format (newlines, speaker labels)
        if i == 0:
            print(f"DEBUG: First transcript snippet (500 chars):\n{transcript[:500]!r}")

        qa_pairs = extract_qa_from_transcript(transcript)
        all_samples.extend(qa_pairs)

        count += 1
        if count % 500 == 0:
            print(f"Processed {count} transcripts | Q/A samples: {len(all_samples)}")

    print(f"\n✅ Total Q/A samples created: {len(all_samples)}")

    if len(all_samples) < 3000:
        print("⚠️ WARNING: Less than 3000 samples. Consider using more data.")

    dataset = Dataset.from_list(all_samples)
    dataset.save_to_disk(OUTPUT_PATH)

    if len(all_samples) == 0:
        raise RuntimeError(
        "❌ No Q/A samples were extracted. "
        "Transcript format mismatch. Check parsing logic."
    )


    print(f"\n📦 Dataset saved to: {OUTPUT_PATH}")
    print("✅ Ready for training.")


if __name__ == "__main__":
    main()
