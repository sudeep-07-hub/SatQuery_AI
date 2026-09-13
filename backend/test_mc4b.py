import asyncio
import sys
import json
import torch
import rasterio
import numpy as np

from mc4b_temporal.tool_adapter import ChangeMambaAdapter

def test_mc4b(file1, file2, query):
    adapter = ChangeMambaAdapter()
    
    with rasterio.open(file1) as src:
        arr1 = src.read(1)
        arr1 = np.nan_to_num(arr1)
        # We'll normalize just to have bounded floats
        arr1 = (arr1 - arr1.min()) / (arr1.max() - arr1.min() + 1e-8)
        # LightweightCNN expects 3 channels usually. If it's 1 channel, we stack it.
        # Actually, let's just make it 3 channels in case.
        arr1 = np.stack([arr1, arr1, arr1], axis=0)
        t1 = torch.tensor(arr1, dtype=torch.float32).unsqueeze(0)
        
    with rasterio.open(file2) as src:
        arr2 = src.read(1)
        arr2 = np.nan_to_num(arr2)
        arr2 = (arr2 - arr2.min()) / (arr2.max() - arr2.min() + 1e-8)
        
        # Ensure sizes match by cropping/padding to arr1 size
        h, w = arr1.shape[1:]
        h2, w2 = arr2.shape
        out2 = np.zeros((h, w), dtype=np.float32)
        min_h, min_w = min(h, h2), min(w, w2)
        out2[:min_h, :min_w] = arr2[:min_h, :min_w]
        
        out2 = np.stack([out2, out2, out2], axis=0)
        t2 = torch.tensor(out2, dtype=torch.float32).unsqueeze(0)
    
    mc1_profile = {
        "image_count": 2,
        "image_1": {"modality": "sar", "gsd_m": 10.0, "crs": "EPSG:32722"},
        "image_2": {"modality": "sar", "gsd_m": 10.0, "crs": "EPSG:32722"},
        "spatial_overlap": 1.0,
        "coregistration_score": 1.0,
    }
    
    result = adapter.execute(mc1_profile, t1, t2, query)
    
    # Don't print the huge change_map tensor, just its shape/info
    if "change_map" in result:
        result["change_map"] = f"List of shape {np.array(result['change_map']).shape}"
    if "binary_change_mask" in result:
        result["binary_change_mask"] = f"List of shape {np.array(result['binary_change_mask']).shape}"
        
    print("Result:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    test_mc4b(sys.argv[1], sys.argv[2], sys.argv[3])
