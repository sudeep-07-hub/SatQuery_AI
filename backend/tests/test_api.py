import requests
import time
import json
from pathlib import Path

BASE_URL = "http://localhost:8000/api"

def run_smoke_test():
    # 1. Submit Query
    print("Submitting query...")
    with open('tests/fixtures/test_geo1.tif', 'rb') as f1, open('tests/fixtures/test_geo2.tif', 'rb') as f2:
        b1 = f1.read()
        b2 = f2.read()
        
    files = [
        ('files', ('test_geo1.tif', b1, 'image/tiff')),
        ('files', ('test_geo2.tif', b2, 'image/tiff'))
    ]
    data = {'query': 'smoke_test Has built-up area increased?'}
    
    response = requests.post(f"{BASE_URL}/query", files=files, data=data)
    response.raise_for_status()
    job_id = response.json()["job_id"]
    print(f"Job ID: {job_id}")
    
    # 2. Poll Status
    print("Polling status...")
    status = ""
    while status not in ["DONE", "FAILED", "ABSTAIN", "INSUFFICIENT_EVIDENCE", "PRECONDITION_FAILED"]:
        res = requests.get(f"{BASE_URL}/jobs/{job_id}/status")
        res.raise_for_status()
        status = res.json()["status"]
        print(f"Status: {status}")
        time.sleep(0.5)
        
    # 3. Get Result
    print(f"\nFinal Status: {status}")
    res = requests.get(f"{BASE_URL}/jobs/{job_id}/result")
    print("Result:")
    print(json.dumps(res.json(), indent=2))
    
    # 4. Get Trace
    res = requests.get(f"{BASE_URL}/jobs/{job_id}/trace")
    print("\nTrace length:", len(res.json()["trace"]))
    
    # 5. Get Evidence Graph
    res = requests.get(f"{BASE_URL}/jobs/{job_id}/evidence_graph")
    eg = res.json().get("evidence_graph")
    if eg:
        print("\nEvidence Graph nodes:", len(eg.get("nodes", [])))

if __name__ == "__main__":
    run_smoke_test()
