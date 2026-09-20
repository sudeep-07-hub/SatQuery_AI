from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os

app = FastAPI(title="SatQuery AI", version="0.2.0")

# Deployed frontends (e.g. a Vercel/Netlify URL) are allowed explicitly via a comma-separated env var:
#   SATQUERY_ALLOWED_ORIGINS=https://satquery.vercel.app,https://satquery.netlify.app
EXTRA_ALLOWED_ORIGINS = [
    origin.strip().rstrip("/")
    for origin in os.getenv("SATQUERY_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=EXTRA_ALLOWED_ORIGINS,
    # Any local dev-server port: Vite moves to 5174, 5175, ... when 5173 is taken
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from mc1.pipeline import run_mc1_pipeline
from job_manager import job_registry, run_agentic_pipeline_sync, build_job_registry, TERMINAL_STATUSES
from qwen.translation import SUPPORTED_QUERY_LANGUAGES, is_supported
from agent.adapters import build_adapters
from qwen import engine_factory
from fastapi import BackgroundTasks, HTTPException

# Optional cap on the combined size of uploaded images per request (0 = unlimited)
MAX_UPLOAD_BYTES = int(float(os.getenv("SATQUERY_MAX_UPLOAD_MB", "0")) * 1024 * 1024)
from fastapi.responses import JSONResponse, FileResponse, Response

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
    query_language: str = Form("en"),
):
    """
    Submits a query to the full MC1->MC8 pipeline.
    Returns a job_id immediately.

    `query_language` is the language `query` is written in (default "en", so clients that do not
    send it behave exactly as before). A non-English query is translated to English before MC2
    sees it, and the transformation is recorded in the job trace as `input_translation`.
    """
    if not is_supported(query_language):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported query_language {query_language!r}; supported: {', '.join(SUPPORTED_QUERY_LANGUAGES)}.",
        )
    # Read files into memory so we don't hold file handles open across async bounds
    files_data = []
    total_bytes = 0
    for f in files:
        content = await f.read()
        total_bytes += len(content)
        if MAX_UPLOAD_BYTES and total_bytes > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Upload too large: the images together exceed {MAX_UPLOAD_BYTES // (1024 * 1024)} MB on this server.",
            )
        files_data.append((f.filename, content))

    job_id = job_registry.create_job()

    # Sync callable → Starlette runs it in a worker thread, keeping the event loop free
    background_tasks.add_task(run_agentic_pipeline_sync, job_id, files_data, query, query_language=query_language)
    return {"job_id": job_id}

@app.get("/api/system")
def get_system_status():
    """Which LLM backend and which specialist engines this server can actually run."""
    backend = engine_factory.resolve_backend()
    llm = {"backend": backend, "model": engine_factory.model_label(backend), "reachable": None}
    if backend == "ollama":
        try:
            import requests
            tags = requests.get(os.getenv("SATQUERY_OLLAMA_URL", "http://localhost:11434") + "/api/tags", timeout=2).json()
            names = {m.get("name") for m in tags.get("models", [])}
            llm["reachable"] = os.getenv("SATQUERY_OLLAMA_MODEL", "qwen3:4b") in names
        except Exception:
            llm["reachable"] = False
    _, tools = build_job_registry("real", build_adapters("real"))
    return {
        "llm": llm,
        "planner_fallback": os.getenv("SATQUERY_PLANNER_FALLBACK", "registry"),
        "tools": tools,
    }

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
    if job["status"] not in TERMINAL_STATUSES:
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

@app.get("/api/jobs/{job_id}/preview/{observation_id}")
def get_observation_preview(job_id: str, observation_id: str):
    """PNG rendering of an uploaded observation (GeoTIFFs cannot be shown by browsers directly)."""
    job = job_registry.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    raster = (job.get("rasters") or {}).get(observation_id)
    if raster is None:
        raise HTTPException(status_code=404, detail="Observation not loaded")
    from raster_io import preview_png_bytes
    return Response(content=preview_png_bytes(raster), media_type="image/png")

@app.get("/api/jobs/{job_id}/export/{format}")
def export_job_result(job_id: str, format: str):
    job = job_registry.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if format == "json_trace":
        format = "json"
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
