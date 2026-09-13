import torch
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

class PaliGemmaVQAAdapter:
    """Unified Adapter for Single-Image Visual Question Answering (Track A)."""

    def __init__(self, model_id: str = "google/paligemma-3b-pt-224"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()

    def predict(self, image_input, query: str) -> dict:
        if isinstance(image_input, str):
            try:
                image = Image.open(image_input).convert("RGB")
            except Exception:
                # Fallback to rasterio for float64 SAR TIFFs
                import rasterio
                import numpy as np
                with rasterio.open(image_input) as src:
                    arr = src.read(1)
                    # Simple min-max normalization to 0-255
                    arr = np.nan_to_num(arr)
                    arr_min, arr_max = arr.min(), arr.max()
                    if arr_max > arr_min:
                        arr = (arr - arr_min) / (arr_max - arr_min) * 255
                    arr = arr.astype(np.uint8)
                    image = Image.fromarray(arr).convert("RGB")
        else:
            image = image_input.convert("RGB")
            
        prompt = f"<image>answer en {query}"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=60,
                return_dict_in_generate=True,
                output_scores=True
            )
        generated_tokens = outputs.sequences[0]
        decoded = self.processor.decode(generated_tokens, skip_special_tokens=True)
        answer = decoded[len(prompt.replace("<image>", "")) :].strip()
        logits = torch.stack(outputs.scores, dim=1)
        probs = torch.softmax(logits, dim=-1)
        token_probs = probs[0, torch.arange(len(outputs.scores)), generated_tokens[-len(outputs.scores):]]
        avg_confidence = float(token_probs.mean().item())
        return {
            "text": answer,
            "evidence_type": "none",
            "spatial_evidence": None,
            "confidence": round(avg_confidence, 4)
        }
