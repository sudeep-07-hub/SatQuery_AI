import os
import sys
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mc4a_vqa.paligemma_adapter import PaliGemmaVQAAdapter
from mc4a_vqa.mock_adapter import MockPaliGemmaVQAAdapter

def run_uncorrected_predict(adapter, image, query):
    """Manually run predict with the incorrect 'question en' prefix to demonstrate the hallucination."""
    prompt = f"<image>question en {query}"
    inputs = adapter.processor(text=prompt, images=image, return_tensors="pt").to(adapter.device)
    with torch.no_grad():
        outputs = adapter.model.generate(
            **inputs,
            max_new_tokens=60,
            return_dict_in_generate=True,
            output_scores=True
        )
    generated_tokens = outputs.sequences[0]
    decoded = adapter.processor.decode(generated_tokens, skip_special_tokens=True)
    answer = decoded[len(prompt.replace("<image>", "")) :].strip()
    return answer

def run_smoke_test():
    print("Attempting to load real PaliGemmaVQAAdapter...")
    try:
        # Check if we should even try downloading based on a local flag or just try and catch
        adapter = PaliGemmaVQAAdapter()
        print(f"Model loaded successfully on device: {adapter.device}")
        
        # Create dummy image
        image = Image.new("RGB", (224, 224), color="green")
        query = "is there a plane?"
        
        print("\n--- Running with UNCORRECTED prefix ('question en') ---")
        try:
            res_uncorrected_text = run_uncorrected_predict(adapter, image, query)
            print("Raw text output:", res_uncorrected_text)
        except Exception as e:
            print(f"Failed uncorrected inference: {e}")
            
        print("\n--- Running with CORRECTED prefix ('answer en') (Using adapter directly) ---")
        try:
            res_corrected = adapter.predict(image, query)
            print("Raw dict output:", res_corrected)
        except Exception as e:
            print(f"Failed corrected inference: {e}")

    except Exception as e:
        print(f"Failed to load real model (likely missing token or disk space): {e}")
        print("\nFalling back to MockPaliGemmaVQAAdapter for CI/Testing...")
        
        adapter = MockPaliGemmaVQAAdapter()
        image = Image.new("RGB", (224, 224), color="green")
        query = "what crop is visible in the first image?"
        
        res_mock = adapter.predict(image, query)
        print(f"\nMock Adapter output for '{query}':", res_mock)

if __name__ == "__main__":
    run_smoke_test()
