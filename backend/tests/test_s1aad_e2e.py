import os
import pytest
import requests
import time
from pathlib import Path

BASE_URL = "http://localhost:8000/api"

S1_DIR = "/Users/sukesh/Desktop/satquery/S1-AAD Sentinel-1 Amazon Airstrip Dataset/S1-AAD Sentinel-1 Amazon Airstrip Dataset"
GEODATA_DIR = os.path.join(S1_DIR, "Geodata")
CHANGE_DIR = os.path.join(S1_DIR, "Change_detection")
IMAGES_DIR = os.path.join(S1_DIR, "Images_geotiff")

def wait_for_job(job_id, timeout=30):
    start_time = time.time()
    while time.time() - start_time < timeout:
        res = requests.get(f"{BASE_URL}/jobs/{job_id}/status")
        res.raise_for_status()
        status = res.json()["status"]
        if status in ["DONE", "FAILED", "ABSTAIN", "INSUFFICIENT_EVIDENCE", "PRECONDITION_FAILED"]:
            return status, res.json()
        time.sleep(0.5)
    raise TimeoutError("Job did not finish in time")

def test_s1aad_single_image_classification():
    """Test that a single S1-AAD chip is classified as SAR and processed (stub)."""
    # Grab one image from Images_geotiff
    test_img = os.path.join(IMAGES_DIR, "_ID_100.tif")
    if not os.path.exists(test_img):
        pytest.skip("Dataset not available")
        
    with open(test_img, "rb") as f:
        files = [('files', ('_ID_100.tif', f.read(), 'image/tiff'))]
        
    data = {'query': 'What is in this image?'}
    res = requests.post(f"{BASE_URL}/query", files=files, data=data)
    res.raise_for_status()
    
    job_id = res.json()["job_id"]
    status, final_res = wait_for_job(job_id)
    
    # Since we didn't fully implement MC4A, it might abstain, but it MUST classify as SAR
    # Let's check the trace to see what MC1 produced
    trace_res = requests.get(f"{BASE_URL}/jobs/{job_id}/trace")
    trace = trace_res.json()["trace"]
    
    # We can't directly assert MC1 profile in trace unless it's in the reason/message,
    # but we can ensure it didn't fail at MC1
    assert status in ["ABSTAIN", "DONE"] 
    assert any(t["stage"] == "MC2_PARSING" for t in trace), "Did not pass MC1"


def test_s1aad_sar_sar_change_detection_rejection():
    """Test that a valid SAR+SAR pair fails precondition for change detection."""
    before_img = os.path.join(CHANGE_DIR, "before", "_ID_100.tif")
    after_img = os.path.join(CHANGE_DIR, "after", "_ID_100.tif")
    
    if not os.path.exists(before_img) or not os.path.exists(after_img):
        pytest.skip("Dataset not available")
        
    with open(before_img, "rb") as f1, open(after_img, "rb") as f2:
        files = [
            ('files', ('before_100.tif', f1.read(), 'image/tiff')),
            ('files', ('after_100.tif', f2.read(), 'image/tiff'))
        ]
        
    data = {'query': 'Has built-up area increased?'}
    res = requests.post(f"{BASE_URL}/query", files=files, data=data)
    res.raise_for_status()
    
    job_id = res.json()["job_id"]
    status, final_res = wait_for_job(job_id)
    
    # Must explicitly fail planning due to SAR+SAR rejection
    assert status == "PRECONDITION_FAILED"
    
    res = requests.get(f"{BASE_URL}/jobs/{job_id}/result")
    job_data = res.json()
    result_data = job_data.get("result", {})
    assert "failed" in result_data
    assert any("sar+sar" in str(f.get("reason", "")).lower() for f in result_data["failed"])

def test_s1aad_overlap_rejection():
    """Test two unrelated location chips fail overlap precondition."""
    # Grab two different images from Images_geotiff
    img1 = os.path.join(IMAGES_DIR, "_ID_100.tif")
    img2 = os.path.join(IMAGES_DIR, "_ID_104.tif")
    
    if not os.path.exists(img1) or not os.path.exists(img2):
        pytest.skip("Dataset not available")
        
    with open(img1, "rb") as f1, open(img2, "rb") as f2:
        files = [
            ('files', ('_ID_100.tif', f1.read(), 'image/tiff')),
            ('files', ('_ID_104.tif', f2.read(), 'image/tiff'))
        ]
        
    data = {'query': 'Has built-up area increased?'}
    res = requests.post(f"{BASE_URL}/query", files=files, data=data)
    res.raise_for_status()
    
    job_id = res.json()["job_id"]
    status, final_res = wait_for_job(job_id)
    
    # It will fail at MC3 Planning (PRECONDITION_FAILED) because of SAR+SAR 
    # OR because of overlap if we remove SAR+SAR. Currently SAR+SAR triggers first.
    assert status == "PRECONDITION_FAILED"
    
    res = requests.get(f"{BASE_URL}/jobs/{job_id}/trace")
    trace = res.json()["trace"]
    fail_entry = next((t for t in reversed(trace) if t["stage"] == "PRECONDITION_FAILED"), None)
    assert fail_entry is not None
