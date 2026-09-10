import torch
import numpy as np
import rasterio
from rasterio.transform import from_origin
from mc4c.engine import MC4CEngine
import os

def create_mock_geotiff(path: str, channels: int):
    # 120x120 dummy data
    data = np.random.randint(0, 10000, (channels, 120, 120), dtype=np.uint16)
    transform = from_origin(0, 0, 10, 10)
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=120,
        width=120,
        count=channels,
        dtype=data.dtype,
        crs='+proj=latlong',
        transform=transform,
    ) as dst:
        dst.write(data)

def test_engine():
    print("Testing MC4CEngine...")
    os.makedirs("mc4c/mock", exist_ok=True)
    opt_path = "mc4c/mock/opt.tif"
    sar_path = "mc4c/mock/sar.tif"
    
    create_mock_geotiff(opt_path, 12)
    create_mock_geotiff(sar_path, 2)
    
    engine = MC4CEngine()
    
    payload = {
        "optical_image": opt_path,
        "sar_image": sar_path,
        "query": "Find the deforested areas",
        "input_profile": {"sensor": "S2", "resolution": 10},
        "task_spec": {"task": "change_detection"}
    }
    
    out = engine.process(payload)
    
    assert "optical_evidence" in out
    assert "sar_evidence" in out
    assert "joint_evidence" in out
    assert "semantic_metadata" in out
    assert "verification_status" in out
    assert "confidence_score" in out
    
    print("MC4CEngine PASSED.")
    print("Verification Status:", out["verification_status"])
    print("Semantic Tags:", out["semantic_metadata"]["optical_tags"])

if __name__ == "__main__":
    test_engine()
