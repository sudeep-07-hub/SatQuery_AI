import asyncio
import sys
import json
import torch
import numpy as np
from PIL import Image

from mc4c.engine import CrossModalEngine

def test_mc4c(optical_path, sar_path, query):
    # Dummy tensors to pass instead of proper rasterio load,
    # because they are PNGs and not georeferenced TIFFs.
    img_opt = Image.open(optical_path).convert("RGB")
    img_sar = Image.open(sar_path).convert("RGB") # usually SAR is 1 channel, but CROMA expects 3 if processed
    
    # Just make dummy tensors to pass the shapes
    t1_opt = torch.rand(1, 3, 224, 224)
    t2_sar = torch.rand(1, 3, 224, 224)
    
    mc1_profile = {
        "image_count": 2,
        "image_1": {"modality": "optical", "filename": "test_optical.png"},
        "image_2": {"modality": "sar", "filename": "test_sar.png"}
    }
    
    engine = CrossModalEngine()
    
    try:
        result = engine.process(t1_opt, t2_sar, query, mc1_profile)
        print("Result:")
        print(result.json(indent=2))
    except Exception as e:
        print("Error:", str(e))

if __name__ == "__main__":
    test_mc4c(sys.argv[1], sys.argv[2], sys.argv[3])
