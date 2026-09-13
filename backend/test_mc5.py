import asyncio
import sys
import json
import os

from job_manager import execute_agentic_pipeline, job_registry

async def test_mc5(filepath, query):
    job_id = job_registry.create_job()
    
    with open(filepath, "rb") as f:
        content = f.read()
        
    files_data = [(os.path.basename(filepath), content)]
    
    await execute_agentic_pipeline(job_id, files_data, query)
    
    job = job_registry.get_job(job_id)
    print("Evidence Objects:")
    print(json.dumps(job.get("evidence_objects", []), indent=2))
    
    print("\nEvidence Graph:")
    print(json.dumps(job.get("evidence_graph", {}), indent=2))
    
if __name__ == "__main__":
    filepath = sys.argv[1]
    query = sys.argv[2]
    asyncio.run(test_mc5(filepath, query))
