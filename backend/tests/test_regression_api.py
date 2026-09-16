"""
Live API regression tests on synthetic GeoTIFFs.
Requires the server (uvicorn main:app on :8000); skipped automatically otherwise.
"""
import os
import time

import pytest
import requests

BASE_URL = "http://localhost:8000/api"
TERMINAL = ["DONE", "FAILED", "ABSTAIN", "INSUFFICIENT_EVIDENCE", "PRECONDITION_FAILED", "MODEL_UNAVAILABLE", "INSUFFICIENT_OBSERVATIONS"]

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")


def create_dummy_tiff(path, bands=3, seed=0):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng = np.random.default_rng(seed)
    transform = from_origin(300000, 4000000, 10, 10)
    with rasterio.open(
        path, 'w', driver='GTiff',
        height=100, width=100,
        count=bands, dtype='uint8',
        crs='+proj=utm +zone=43 +datum=WGS84',
        transform=transform,
    ) as dst:
        for i in range(1, bands + 1):
            dst.write(rng.integers(0, 255, (100, 100), dtype=np.uint8), i)


def wait_for_job(job_id, timeout=600):
    start_time = time.time()
    while time.time() - start_time < timeout:
        status = requests.get(f"{BASE_URL}/jobs/{job_id}/status").json()["status"]
        if status in TERMINAL:
            return status
        time.sleep(1)
    raise TimeoutError("Job did not finish in time")


@pytest.fixture(autouse=True, scope="module")
def setup_test_data():
    create_dummy_tiff(os.path.join(TEST_DATA_DIR, "optical_1.tif"), bands=3, seed=1)
    create_dummy_tiff(os.path.join(TEST_DATA_DIR, "optical_2.tif"), bands=3, seed=2)


def test_api_regression_unsupported_format():
    files = [('files', ('test.txt', b'hello world', 'text/plain'))]
    res = requests.post(f"{BASE_URL}/query", files=files, data={'query': 'test'})
    res.raise_for_status()
    job_id = res.json()["job_id"]
    assert wait_for_job(job_id) == "PRECONDITION_FAILED"
    result = requests.get(f"{BASE_URL}/jobs/{job_id}/result").json()["result"]
    assert any("Unsupported extension" in f["reason"] for f in result["failed"])


def test_api_smoke_test_keyword_does_not_bypass_mc1():
    """'smoke_test' in a real API query must not replace MC1's verdict with a synthetic profile."""
    files = [('files', ('fake.tif', b'not really a tiff', 'image/tiff'))]
    res = requests.post(f"{BASE_URL}/query", files=files, data={'query': 'smoke_test: did it change?'})
    job_id = res.json()["job_id"]
    assert wait_for_job(job_id) == "PRECONDITION_FAILED"


def test_api_regression_optical_pair_change_and_exports():
    with open(os.path.join(TEST_DATA_DIR, "optical_1.tif"), "rb") as f1, \
         open(os.path.join(TEST_DATA_DIR, "optical_2.tif"), "rb") as f2:
        files = [
            ('files', ('opt1.tif', f1.read(), 'image/tiff')),
            ('files', ('opt2.tif', f2.read(), 'image/tiff'))
        ]
    res = requests.post(f"{BASE_URL}/query", files=files, data={'query': 'What changed between these two images?'})
    job_id = res.json()["job_id"]
    assert wait_for_job(job_id) == "DONE"

    final_result = requests.get(f"{BASE_URL}/jobs/{job_id}/result").json()["result"]
    assert final_result["change_statistics"] is not None
    assert final_result["planner"]["tool_selection"] in ("qwen3", "registry_rules")

    for fmt in ("json", "geojson", "pdf", "png"):
        assert requests.get(f"{BASE_URL}/jobs/{job_id}/export/{fmt}").status_code == 200
