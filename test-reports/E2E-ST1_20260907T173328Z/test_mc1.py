import asyncio
import json
import os
import sys
import shutil
import io

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))

from fastapi import UploadFile
from starlette.datastructures import Headers
from mc1.pipeline import run_mc1_pipeline

def create_mock_upload_file(filename, filepath):
    with open(filepath, "rb") as f:
        content = f.read()
    file_obj = io.BytesIO(content)
    return UploadFile(filename=filename, file=file_obj, size=len(content))

async def test_mc1():
    base_dir = "/Users/sukesh/Desktop/satquery/S1-AAD Sentinel-1 Amazon Airstrip Dataset/S1-AAD Sentinel-1 Amazon Airstrip Dataset"
    
    print("--- Test 1: Valid single-image chip ---")
    try:
        f1 = create_mock_upload_file("_ID_100.tif", f"{base_dir}/Images_geotiff/_ID_100.tif")
        profile = await run_mc1_pipeline([f1], "What is this?")
        print("PASS: Profile generated")
        print(json.dumps(profile, indent=2))
    except Exception as e:
        print("FAIL:", e)

    print("\n--- Test 2: Valid bi-temporal pair ---")
    try:
        f1 = create_mock_upload_file("before_ID_100.tif", f"{base_dir}/Change_detection/before/_ID_100.tif")
        f2 = create_mock_upload_file("after_ID_100.tif", f"{base_dir}/Change_detection/after/_ID_100.tif")
        profile = await run_mc1_pipeline([f1, f2], "Did it change?")
        print("PASS: Profile generated")
        print(json.dumps(profile, indent=2))
    except Exception as e:
        print("FAIL:", e)

    print("\n--- Test 3: Non-overlapping footprint case ---")
    try:
        # Use _ID_100 and _ID_104 from single images, they are different footprints
        f1 = create_mock_upload_file("_ID_100.tif", f"{base_dir}/Images_geotiff/_ID_100.tif")
        f2 = create_mock_upload_file("_ID_104.tif", f"{base_dir}/Images_geotiff/_ID_104.tif")
        profile = await run_mc1_pipeline([f1, f2], "Did it change?")
        if not profile.get("task_executable"):
            print("PASS: Rejected correctly ->", profile["warnings"])
        else:
            print("FAIL: Silently accepted non-overlapping pair. Profile:")
            print(json.dumps(profile, indent=2))
    except Exception as e:
        print("FAIL:", type(e), e)

    print("\n--- Test 4: Mismatched CRS case ---")
    import rasterio
    import shutil
    # We will synthesize this by taking _ID_100.tif and changing its CRS in metadata
    shutil.copy(f"{base_dir}/Change_detection/after/_ID_100.tif", "/tmp/mismatched_crs.tif")
    with rasterio.open("/tmp/mismatched_crs.tif", "r+") as src:
        src.crs = rasterio.crs.CRS.from_epsg(4326)
    
    try:
        f1 = create_mock_upload_file("before_ID_100.tif", f"{base_dir}/Change_detection/before/_ID_100.tif")
        f2 = create_mock_upload_file("mismatched_crs.tif", "/tmp/mismatched_crs.tif")
        profile = await run_mc1_pipeline([f1, f2], "Did it change?")
        if not profile.get("task_executable"):
            print("PASS: Rejected correctly ->", profile["warnings"])
        else:
            print("FAIL: Silently accepted CRS mismatch or handled it. Profile:")
            print(json.dumps(profile, indent=2))
    except Exception as e:
        print("FAIL:", type(e), e)

    print("\n--- Test 5: Corrupted file case ---")
    with open("/tmp/corrupted.tif", "w") as f:
        f.write("This is not a valid TIFF file at all.")
    try:
        f1 = create_mock_upload_file("corrupted.tif", "/tmp/corrupted.tif")
        profile = await run_mc1_pipeline([f1], "What is this?")
        if not profile.get("task_executable"):
            print("PASS: Rejected correctly ->", profile["warnings"])
        else:
            print("FAIL: Should have rejected corrupted file")
    except Exception as e:
        print("FAIL:", type(e), e)

if __name__ == "__main__":
    asyncio.run(test_mc1())
