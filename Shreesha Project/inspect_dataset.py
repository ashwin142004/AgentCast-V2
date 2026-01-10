from datasets import load_dataset

ds = load_dataset(
    "Whispering-GPT/lex-fridman-podcast-transcript-audio",
    split="train",
    verification_mode="no_checks"
)

# remove audio so it doesn't crash
# The column is named 'text', not 'transcript'.
if "audio" in ds.column_names:
    ds = ds.remove_columns(["audio"])

print("Columns:", ds.column_names)
print("\n--- SAMPLE TRANSCRIPT ---\n")
print(ds[0]["text"][:2000])
