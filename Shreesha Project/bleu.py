from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import evaluate

MODEL_NAME = "google/gemma-2b"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")

gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,
    temperature=0.7,
    do_sample=True
)

prompt = input("Enter your prompt: ")

print("\nGenerating response...")
outputs = gen_pipeline(prompt)
candidate = outputs[0]["generated_text"]

print("\nModel Response:\n", candidate)

bleu = evaluate.load("bleu")
reference = [["Alan Turing was a British mathematician and computer scientist. He is widely considered the father of theoretical computer science and artificial intelligence. He developed the Turing machine and played a key role in breaking German ciphers during World War II."]]
results = bleu.compute(predictions=[candidate], references=reference)

print(f"\nBLEU Score: {results['bleu']:.4f}")
