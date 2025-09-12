
from huggingface_hub import snapshot_download
from vllm import LLM
import os

HUGGINGFACE_TOKEN = "abc123"
os.environ['HF_TOKEN'] = HUGGINGFACE_TOKEN

local_model_path = snapshot_download(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    cache_dir="/scratch/akranjcevic/models/llama3.1-8b-instruct"
)
