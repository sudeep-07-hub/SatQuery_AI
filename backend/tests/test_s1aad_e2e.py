"""
Live API tests against real S1-AAD Sentinel-1 data.
Requires the server (uvicorn main:app on :8000); skipped automatically otherwise.
"""
import os
import time

import pytest
import requests

BASE_URL = "http://localhost:8000/api"
TERMINAL = ["DONE", "FAILED", "ABSTAIN", "INSUFFICIENT_EVIDENCE", "PRECONDITION_FAILED", "MODEL_UNAVAILABLE", "INSUFFICIENT_OBSERVATIONS"]

S1_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "S1-AAD Sentinel-1 Amazon Airstrip Dataset",
    "S1-AAD Sentinel-1 Amazon Airstrip Dataset",
)
CHANGE_DIR = os.path.join(S1_DIR, "Change_detection")
IMAGES_DIR = os.path.join(S1_DIR, "Images_geotiff")


def submit(paths_and_names, query):
    files = [("files", (name, open(path, "rb").read(), "image/tiff")) for path, name in paths_and_names]
    res = requests.post(f"{BASE_URL}/query", files=files, data={"query": query})
    res.raise_for_status()
    return res.json()["job_id"]


def wait_for_job(job_id, timeout=600):
    start_time = time.time()
    while time.time() - start_time < timeout:
        status = requests.get(f"{BASE_URL}/jobs/{job_id}/status").json()["status"]
        if status in TERMINAL:
            return status
        time.sleep(1)
    raise TimeoutError("Job did not finish in time")


def test_s1aad_sar_pair_change_detection_runs_on_real_pixels():
    """A genuine before/after SAR pair is analysed by an available change engine (not rejected, not fabricated)."""
    before = os.path.join(CHANGE_DIR, "before", "_ID_32.tif")
    after = os.path.join(CHANGE_DIR, "after", "_ID_32.tif")
    if not (os.path.exists(before) and os.path.exists(after)):
        pytest.skip("Dataset not available")

    job_id = submit([(before, "before_32.tif"), (after, "after_32.tif")], "What changed between these two images?")
    assert wait_for_job(job_id) == "DONE"

    result = requests.get(f"{BASE_URL}/jobs/{job_id}/result").json()["result"]
    stats = result["change_statistics"]
    assert stats["region_count"] > 0
    assert stats["changed_area_m2"] > 0
    assert any("upload order" in c for c in result["caveats"]), "missing-date assumption must be disclosed"

    evidence = requests.get(f"{BASE_URL}/jobs/{job_id}/structured_trace").json()["structured_trace"]["evidence_objects"]
    assert evidence and all(ev["source_model"].startswith("CLASSICAL_CHANGE_DETECTION") for ev in evidence)
    lat = result["change_overlay"]["bounds_wgs84"][0][0]
    assert -35 < lat < 10, "overlay must be reprojected to WGS84 (Amazon latitudes)"

    for fmt in ("json", "json_trace", "geojson", "pdf", "png"):
        assert requests.get(f"{BASE_URL}/jobs/{job_id}/export/{fmt}").status_code == 200
    assert requests.get(f"{BASE_URL}/jobs/{job_id}/preview/image_1").headers["content-type"] == "image/png"


def test_s1aad_overlap_rejection():
    """Two chips of different airstrips do not overlap and must be rejected at MC1."""
    img1 = os.path.join(IMAGES_DIR, "_ID_100.tif")
    img2 = os.path.join(IMAGES_DIR, "_ID_104.tif")
    if not (os.path.exists(img1) and os.path.exists(img2)):
        pytest.skip("Dataset not available")

    job_id = submit([(img1, "_ID_100.tif"), (img2, "_ID_104.tif")], "What changed between these images?")
    assert wait_for_job(job_id) == "PRECONDITION_FAILED"

    result = requests.get(f"{BASE_URL}/jobs/{job_id}/result").json()["result"]
    assert any("overlap" in f["reason"].lower() for f in result["failed"])
