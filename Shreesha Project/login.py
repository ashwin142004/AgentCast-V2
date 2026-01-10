from huggingface_hub import login
import os

login(token=os.getenv("hg_token"))
