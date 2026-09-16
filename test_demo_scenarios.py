import asyncio
import sys
from backend.job_manager import execute_agentic_pipeline, job_registry

async def run_scenario(scenario_name, query, expected_status="DONE", files=None):
    print(f"\n{'='*60}")
    print(f"Executing Scenario: {scenario_name}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    
    if files is None:
        files = [("image1.tif", b"dummy"), ("image2.tif", b"dummy")]
        
    job_id = job_registry.create_job()
    
    await execute_agentic_pipeline(job_id, files, query, execution_mode="fixture")
    
    job = job_registry.get_job(job_id)
    result = job.get("result", {})
    status = job.get("status")
    
    if status == "FAILED":
        for entry in job.get("progress_trace", []):
            if entry["stage"] == "FAILED":
                print(f"ERROR: {entry.get('error')}")
                print(f"TRACEBACK: {entry.get('traceback')}")
                
    print(f"\nFinal Job Status: {status}")
    if result:
        print(f"Execution Status: {result.get('execution_status')}")
        print(f"Final Answer: {result.get('final_answer')}")
        print(f"Evidence Found: {len(result.get('claims', []))} claims")
        if result.get("caveats"):
            print(f"Caveats: {result.get('caveats')}")
            
    if status == expected_status or (expected_status == "FAILED" and status != "DONE"):
        print(f"✅ SCENARIO {scenario_name.split()[0]} PASSED")
    else:
        print(f"❌ SCENARIO {scenario_name.split()[0]} FAILED (Expected {expected_status}, got {status})")

async def main():
    await run_scenario("D1 - Single-image VQA", "smoke_test: Are there any buildings in the image?", "DONE", files=[("image1.tif", b"dummy")])
    await run_scenario("D2 - Multi-tool Orchestration", "smoke_test: compound query", "DONE")
    # The only fusion-capable engine fails and no alternative exists, so the honest outcome is no answer
    await run_scenario("D3 - Specialist failure/recovery", "smoke_test: fail optical sar fusion", "INSUFFICIENT_EVIDENCE")
    await run_scenario("D4 - Invalid-input Rejection", "Check this image for buildings.", "PRECONDITION_FAILED", files=[("image1.tif", b"dummy")])
    
if __name__ == "__main__":
    asyncio.run(main())
