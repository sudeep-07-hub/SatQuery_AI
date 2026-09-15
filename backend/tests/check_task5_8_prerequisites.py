import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM

def check_qwen3_availability():
    target_model_id = "Qwen/Qwen3-4B-Instruct-2507"
    
    print("Checking Qwen3-4B availability...")
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(target_model_id, trust_remote_code=True)
        print("Model configuration is accessible.")
        return True
    except Exception as e:
        print(f"FAILED to access Qwen3-4B: {e}")
        return False

import os
import glob
def check_dataset_scale():
    cache_dir = "/Users/sukesh/Desktop/satquery/backend/data/features"
    samples = glob.glob(os.path.join(cache_dir, "sample_*.pt"))
    print(f"Found {len(samples)} physical visual samples in cache.")
    return len(samples)

if __name__ == "__main__":
    qwen_available = check_qwen3_availability()
    sample_count = check_dataset_scale()
    
    print("\n--- STATUS ---")
    print(f"Qwen Available: {qwen_available}")
    print(f"Sample Count: {sample_count}")
