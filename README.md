# SatQuery AI

Ask a question in plain English about satellite imagery, and get an answer backed by
traceable evidence — not just a model output.

Upload one or two scenes (optical or SAR), type a query such as *"Has the forest here been
cleared between these two dates?"*, and SatQuery plans which analysis tools to run, runs them,
assembles the findings into an evidence graph, checks that graph for internal conflicts, and
returns an answer with a confidence level and a full execution trace. Every claim in the answer
can be followed back to the tool run and the pixels that produced it.

---

## How it works

A query moves through a pipeline of mission components (MCs). Each stage has a typed contract,
so a stage can be swapped or disabled without breaking the rest.

| Stage | Module | Responsibility |
|---|---|---|
| **MC1** | [backend/mc1/](backend/mc1/) | Qualify the input: format validation, metadata extraction, modality classification (optical/SAR), spatial and temporal analysis, and a hard/soft compatibility profile |
| **MC2** | [backend/qwen/](backend/qwen/) | Understand the query and decompose it into a structured `TaskSpec` |
| **MC3** | [backend/mc3_planner/](backend/mc3_planner/) | Tool Registry, workflow planner, and dispatcher — decide *which* engines can answer this question given these inputs |
| **MC4a** | [backend/mc4a_vqa/](backend/mc4a_vqa/) | Single-image visual question answering and captioning (PaliGemma) |
| **MC4b** | [backend/mc4b_temporal/](backend/mc4b_temporal/) | Temporal change analysis — a learned backbone plus a classical log-ratio / CVA engine |
| **MC4c** | [backend/mc4c/](backend/mc4c/) | Optical–SAR cross-modal fusion (CROMA) with query-conditioned attention |
| **MC5** | [backend/mc5_evidence/](backend/mc5_evidence/) | Normalise every tool's output into a common evidence graph |
| **MC6** | [backend/mc6_verification/](backend/mc6_verification/) | Verify the graph: conflict detection and temporal consistency checks |
| **MC8** | [backend/mc8_export/](backend/mc8_export/) | Export the result as JSON, GeoJSON, or PDF |

Orchestration lives in [backend/agent/](backend/agent/) (planning, tool binding, recovery) and
[backend/controller/](backend/controller/); job lifecycle in
[backend/job_manager.py](backend/job_manager.py), which runs pipelines on a worker thread so the
API stays responsive.

**Query intelligence is graceful.** Intent normally comes from Qwen3 (`qwen3:4b` via Ollama).
When Qwen3 is unreachable or cannot resolve the intent, the system falls back to deterministic
Tool Registry rules — and the trace says so explicitly (`planner.query_intelligence =
registry_rules`) rather than silently pretending the LLM ran.

### Registered tools

Single Image VQA & Captioning · Temporal Change Analysis · Classical Change Detection
(log-ratio / CVA) · Optical–SAR Multimodal Fusion

---

## Running it

### Backend

```bash
cd backend
pip install -r requirements.txt
python3 -m uvicorn main:app --port 8000
```

`GET /api/system` reports which LLM backend and which specialist engines this particular
server can actually run — useful to confirm what a given deployment will do before demoing it.

### Frontend

```bash
cd frontend
npm install
npm run dev            # http://localhost:5173
```

See [frontend/README.md](frontend/README.md) for routes, configuration, and production builds.

### Tests

```bash
cd backend
python3 -m pytest        # 739 tests
```

Tests that need a live API server are skipped automatically when nothing is listening on
`localhost:8000`.

---

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/validate` | MC1 validation only — check inputs before committing to a full run |
| `POST /api/query` | Submit a query to the full pipeline; returns a `job_id` immediately |
| `GET /api/system` | LLM backend and available engines for this deployment |
| `GET /api/jobs/{id}/status` | Job status |
| `GET /api/jobs/{id}/result` | Final answer |
| `GET /api/jobs/{id}/trace` | Execution trace |
| `GET /api/jobs/{id}/evidence_graph` | Evidence graph |
| `GET /api/jobs/{id}/structured_trace` | Structured step-by-step trace |
| `GET /api/jobs/{id}/preview/{observation_id}` | PNG preview of an observation |
| `GET /api/jobs/{id}/export/{format}` | Export as `json`, `geojson`, or `pdf` |

### Configuration

| Variable | Default | Meaning |
|---|---|---|
| `SATQUERY_QWEN_BACKEND` | `ollama` | LLM backend: `ollama` (Qwen3-4B Q4_K_M, ~2.5 GB), `transformers` (bf16, ~8 GB RAM), or `none` (registry rules only) |
| `SATQUERY_OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint |
| `SATQUERY_OLLAMA_MODEL` | `qwen3:4b` | Ollama model tag |
| `SATQUERY_PLANNER_FALLBACK` | `registry` | Planner fallback strategy |
| `SATQUERY_DISABLED_TOOLS` | *(empty)* | Comma-separated tools to disable on memory-constrained hosts |
| `SATQUERY_ALLOWED_ORIGINS` | *(empty)* | Extra CORS origins; `localhost` is always allowed |
| `SATQUERY_MAX_UPLOAD_MB` | `0` (unlimited) | Cap on combined upload size per request |
| `SATQUERY_MAX_JOBS` | `0` (unlimited) | Cap on retained jobs; oldest finished jobs are evicted first |

---

## Deployment

[render.yaml](render.yaml) is a Render blueprint for the backend on the free tier
(512 MB RAM, 0.1 CPU). That plan runs Tool Registry rules, classical change detection, and
exports; PaliGemma VQA and CROMA fusion are disabled there because they need gigabytes of RAM.
The frontend deploys to Vercel with `VITE_API_BASE` pointed at the backend URL.

## Data & model weights

Not committed — they are gigabytes, and some carry unresolved dataset licences:

- **CROMA weights** → `backend/download_croma.py` → `backend/mc4c/weights/`
- **BigEarthNet** → `backend/download_bigearthnet.py`
- **Feature caches** → `backend/scripts/` regenerates `backend/data/features/`

Evaluation harness and fixtures: [backend/evaluation/](backend/evaluation/).
Adaptation-run results: [backend/data/adaptation/metrics/](backend/data/adaptation/metrics/).

## Known limitations

[KNOWN_GAPS.md](KNOWN_GAPS.md) records what does not work yet and why — ambiguous target
resolution across two images, omitted evidence-graph edges, and others. It is deliberately
kept current rather than pruned.
