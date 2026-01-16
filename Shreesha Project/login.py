from huggingface_hub import login
import os
from dotenv import load_dotenv

load_dotenv()

login(token=os.getenv("hg_token"))
