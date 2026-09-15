import asyncio
from job_manager import job_registry, execute_agentic_pipeline
import traceback

async def run():
    job_id = job_registry.create_job()
    await execute_agentic_pipeline(job_id, [("a.tif", b"")], "What is in this image? smoke_test", execution_mode="fixture")
    job = job_registry.get_job(job_id)
    print("STATUS:", job["status"])
    for t in job["progress_trace"]:
        print(t)
    if job.get("result"):
        print("RESULT:", job["result"])

asyncio.run(run())
