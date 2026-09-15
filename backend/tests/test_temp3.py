import asyncio
from job_manager import execute_agentic_pipeline, job_registry

async def run():
    job_id = job_registry.create_job()
    await execute_agentic_pipeline(job_id, [("file1", b""), ("file2", b"")], "change detection smoke_test")
    job = job_registry.get_job(job_id)
    print("STATUS:", job["status"])
    for t in job["progress_trace"]:
        print(t)
    if job.get("result"):
        print("RESULT:", job["result"])

asyncio.run(run())
