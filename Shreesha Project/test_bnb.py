import torch
try:
    import bitsandbytes as bnb
    print("bitsandbytes imported successfully")
except ImportError as e:
    print(f"bitsandbytes validation failed: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
