import os
import sys
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mc4a_vqa.paligemma_adapter import PaliGemmaVQAAdapter

def run_smoke_test():
    print("Initializing PaliGemmaVQAAdapter...")
    try:
        adapter = PaliGemmaVQAAdapter()
        print(f"Model loaded successfully on device: {adapter.device}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Create dummy image
    image = Image.new("RGB", (224, 224), color="green")
    query = "is there a plane?"
    
    print("\n--- Running with UNCORRECTED prefix ('question en') ---")
    try:
        res_uncorrected = adapter.predict(image, query, prompt_prefix="question en")
        print("Raw output:", res_uncorrected)
    except Exception as e:
        print(f"Failed uncorrected inference: {e}")
        
    print("\n--- Running with CORRECTED prefix ('answer en') ---")
    try:
        res_corrected = adapter.predict(image, query, prompt_prefix="answer en")
        print("Raw output:", res_corrected)
    except Exception as e:
        print(f"Failed corrected inference: {e}")

if __name__ == "__main__":
    if "HUGGING_FACE_HUB_TOKEN" not in os.environ:
        print("WARNING: HUGGING_FACE_HUB_TOKEN is not set in the environment.")
    
    run_smoke_test()
