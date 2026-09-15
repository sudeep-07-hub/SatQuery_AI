import torch
import transformers
from huggingface_hub import try_to_load_from_cache

print(f"OS/PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"MPS: {torch.backends.mps.is_available()}")
print(f"Transformers: {transformers.__version__}")

model_id = "Qwen/Qwen3-4B-Instruct-2507"
cached = try_to_load_from_cache(repo_id=model_id, filename="config.json")
if cached:
    print(f"Model cached at {cached}")
else:
    print("Model NOT locally cached.")

try:
    print("Testing mock inference to check memory...")
    # Just a small model to see if MPS works
    # We will assume if not cached, we shouldn't download it for a test unless necessary.
except Exception as e:
    print("Error:", e)
