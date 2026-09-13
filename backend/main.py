from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os

app = FastAPI(title="SatQuery AI", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from mc1.pipeline import run_mc1_pipeline
from job_manager import job_registry, execute_agentic_pipeline
import asyncio
from fastapi import BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse

@app.post("/api/validate")
async def validate(
    files: List[UploadFile] = File(...),
    query: str = Form(...),
):
    """
    Stage 1 (MC1) validation endpoint.
    Executes a 12-step validation pipeline on the inputs.
    """
    profile = await run_mc1_pipeline(files, query)
    return profile

@app.post("/api/query")
async def submit_query(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    query: str = Form(...),
):
    """
    Submits a query to the full MC1->MC7 pipeline.
    Returns a job_id immediately.
    """
    job_id = job_registry.create_job()
    
    # Read files into memory so we don't hold file handles open across async bounds
    # In a production system, we'd save to a temp dir or object storage
    files_data = []
    for f in files:
        content = await f.read()
        files_data.append((f.filename, content))
        
    background_tasks.add_task(execute_agentic_pipeline, job_id, files_data, query)
    return {"job_id": job_id}

@app.get("/api/jobs/{job_id}/status")
def get_job_status(job_id: str):
    job = job_registry.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "status": job["status"]}

@app.get("/api/jobs/{job_id}/result")
def get_job_result(job_id: str):
    job = job_registry.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] not in ["DONE", "FAILED", "ABSTAIN", "INSUFFICIENT_EVIDENCE", "PRECONDITION_FAILED"]:
        return JSONResponse(status_code=202, content={"message": "Job still processing"})
    return {"job_id": job_id, "status": job["status"], "result": job.get("result")}

@app.get("/api/jobs/{job_id}/trace")
def get_job_trace(job_id: str):
    job = job_registry.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "trace": job.get("progress_trace", [])}

@app.get("/api/jobs/{job_id}/evidence_graph")
def get_job_evidence_graph(job_id: str):
    job = job_registry.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "evidence_graph": job.get("evidence_graph")}

@app.get("/api/jobs/{job_id}/structured_trace")
def get_job_structured_trace(job_id: str):
    job = job_registry.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job_id, 
        "structured_trace": {
            "evidence_objects": job.get("evidence_objects", []),
            "evidence_graph": job.get("evidence_graph", {})
        }
    }

@app.get("/api/jobs/{job_id}/export/{format}")
def export_job_result(job_id: str, format: str):
    job = job_registry.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    if format not in job.get("exports", {}):
        raise HTTPException(status_code=404, detail=f"Export format '{format}' not available or not generated yet")
        
    filepath = job["exports"][format]
    filename = os.path.basename(filepath)
    
    media_types = {
        "json": "application/json",
        "geojson": "application/geo+json",
        "pdf": "application/pdf",
        "png": "image/png"
    }
    media_type = media_types.get(format, "application/octet-stream")
    
    return FileResponse(
        filepath, 
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
