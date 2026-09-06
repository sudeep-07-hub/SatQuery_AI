import os
import pytest
import requests
import time
from pathlib import Path

BASE_URL = "http://localhost:8000/api"

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")

def create_dummy_tiff(path, bands=3):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    os.makedirs(os.path.dirname(path), exist_ok=True)
    transform = from_origin(300000, 4000000, 10, 10)
    with rasterio.open(
        path, 'w', driver='GTiff',
        height=100, width=100,
        count=bands, dtype='uint8',
        crs='+proj=utm +zone=43 +datum=WGS84',
        transform=transform,
    ) as dst:
        for i in range(1, bands + 1):
            dst.write(np.random.randint(0, 255, (100, 100), dtype=np.uint8), i)

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

@pytest.fixture(autouse=True, scope="session")
def setup_test_data():
    create_dummy_tiff(os.path.join(TEST_DATA_DIR, "optical_1.tif"), bands=3)
    create_dummy_tiff(os.path.join(TEST_DATA_DIR, "optical_2.tif"), bands=3)
    create_dummy_tiff(os.path.join(TEST_DATA_DIR, "sar_1.tif"), bands=1)

def test_api_regression_single_optical():
    with open(os.path.join(TEST_DATA_DIR, "optical_1.tif"), "rb") as f:
        files = [('files', ('optical_1.tif', f.read(), 'image/tiff'))]
    data = {'query': 'Find the building'}
    res = requests.post(f"{BASE_URL}/query", files=files, data=data)
    res.raise_for_status()
    job_id = res.json()["job_id"]
    status, _ = wait_for_job(job_id)
    assert status in ["DONE", "ABSTAIN"] # Abstain if no tool is registered for single optical, which is currently the case (only ChangeMamba)

def test_api_regression_optical_pair_smoke_test():
    with open(os.path.join(TEST_DATA_DIR, "optical_1.tif"), "rb") as f1, \
         open(os.path.join(TEST_DATA_DIR, "optical_2.tif"), "rb") as f2:
        files = [
            ('files', ('optical_1.tif', f1.read(), 'image/tiff')),
            ('files', ('optical_2.tif', f2.read(), 'image/tiff'))
        ]
    data = {'query': 'smoke_test: did it change?'}
    res = requests.post(f"{BASE_URL}/query", files=files, data=data)
    res.raise_for_status()
    job_id = res.json()["job_id"]
    status, result = wait_for_job(job_id)
    
    assert status == "DONE"
    res = requests.get(f"{BASE_URL}/jobs/{job_id}/result")
    final_result = res.json()["result"]
    assert "change_map" in final_result
    assert "change_statistics" in final_result

def test_api_regression_unsupported_format():
    files = [('files', ('test.txt', b'hello world', 'text/plain'))]
    data = {'query': 'test'}
    res = requests.post(f"{BASE_URL}/query", files=files, data=data)
    res.raise_for_status()
    job_id = res.json()["job_id"]
    status, _ = wait_for_job(job_id)
    assert status == "FAILED"
    
def test_api_regression_exports():
    with open(os.path.join(TEST_DATA_DIR, "optical_1.tif"), "rb") as f1, \
         open(os.path.join(TEST_DATA_DIR, "optical_2.tif"), "rb") as f2:
        files = [
            ('files', ('opt1.tif', f1.read(), 'image/tiff')),
            ('files', ('opt2.tif', f2.read(), 'image/tiff'))
        ]
    data = {'query': 'smoke_test: changed?'}
    res = requests.post(f"{BASE_URL}/query", files=files, data=data)
    job_id = res.json()["job_id"]
    status, _ = wait_for_job(job_id)
    assert status == "DONE"
    
    # Test JSON export
    res = requests.get(f"{BASE_URL}/jobs/{job_id}/export/json")
    assert res.status_code == 200
    
    # Test GeoJSON export
    res = requests.get(f"{BASE_URL}/jobs/{job_id}/export/geojson")
    assert res.status_code == 200
    
    # Test PDF export
    res = requests.get(f"{BASE_URL}/jobs/{job_id}/export/pdf")
    assert res.status_code == 200
    
    # Test PNG export
    res = requests.get(f"{BASE_URL}/jobs/{job_id}/export/png")
    assert res.status_code == 200
