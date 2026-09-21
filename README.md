<div align="center">

# 🛰️ SatQuery AI

### Agentic Assistant for Satellite Imagery

*Ask in plain English. Get an answer you can audit, all the way down to the pixels.*

[![LIVE DEMO](https://img.shields.io/badge/🌐_LIVE-DEMO-F2600C?style=for-the-badge)](https://sat-query-ai-mu.vercel.app/)
[![PIPELINE](https://img.shields.io/badge/🧩_PIPELINE-MC1→MC8-blue?style=for-the-badge)](#system-architecture)
[![LLM](https://img.shields.io/badge/🧠_PLANNER-QWEN3--4B-purple?style=for-the-badge)](https://ollama.com/library/qwen3)
[![TESTS](https://img.shields.io/badge/✅_TESTS-739-green?style=for-the-badge)](#tests)

---

</div>

## Project Overview

**SatQuery AI** turns a plain-English question about satellite imagery into an evidence-backed
answer. Upload one or two scenes — optical or SAR — ask something like *"Has a new airstrip been
cleared between these two acquisitions?"*, and the system decides which analysis engines can
answer it, runs them, normalises their outputs into an evidence graph, verifies that graph for
internal conflicts, and returns an answer with its confidence, its caveats, and a full execution
trace.

The point is not that a model produced an answer. The point is that **every claim can be followed
back to the tool run and the pixels that produced it** — and that when the system cannot answer,
it says so instead of guessing.

<br>

## Key Features

**Evidence-Grounded Answers** — every claim traces to a tool run, a region, and a source image

**Honest Capability Reporting** — `GET /api/system` tells you which engines a deployment can
actually run, and the UI disables what it cannot do

**Graceful Query Intelligence** — Qwen3 plans the run; when it is unreachable, deterministic Tool
Registry rules take over and the trace records that the LLM did not run

**Auditable Execution Trace** — every stage, tool call, and verification decision is inspectable

**Abstention Over Fabrication** — insufficient evidence returns `ABSTAIN`, never a plausible guess

**Spatial Evidence on a Map** — change regions drawn on Leaflet, tied to the claims that cite them

**Before/After Comparison** — drag-divider swipe between the two acquisitions

**Conflict Verification** — MC6 checks the evidence graph before anything reaches the answer

**Standards-Friendly Export** — JSON trace, GeoJSON regions, or a PDF report (written in English; see below)

**Multilingual Interface** — English, हिन्दी, ಕನ್ನಡ, తెలుగు and தமிழ், with the script's font loaded
only when that language is chosen. Ask in any of them: the query is translated to English before
the pipeline plans anything, and the trace shows both the question as asked and the English the
pipeline actually reasoned over

**Voice Input** — dictate a question with the browser's own speech recognition; the transcript
lands editable in the composer and nothing is sent until you press Send

<br>

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       SATQUERY PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   MC1          MC2            MC3              MC4               │
│  ┌──────┐   ┌────────┐   ┌───────────┐   ┌──────────────┐        │
│  │Input │──▶│ Query  │──▶│   Tool    │──▶│  Specialist  │        │
│  │Qualify│  │ Intel  │   │ Registry  │   │   Engines    │        │
│  └──────┘   └────────┘   └───────────┘   └──────┬───────┘        │
│   format     Qwen3 →      which engine    4a VQA│ 4b change      │
│   modality   TaskSpec     can answer?     4c fusion              │
│   georef     (or rules)   given inputs           │               │
│                                                  ▼               │
│   MC8            MC6              MC5      ┌──────────┐          │
│  ┌──────┐   ┌───────────┐   ┌──────────┐   │  Tool    │          │
│  │Export│◀──│  Verify   │◀──│ Evidence │◀──│ Outputs  │          │
│  └──────┘   └───────────┘   └──────────┘   └──────────┘          │
│  json/geo    conflicts &     normalise                           │
│  /pdf        consistency     to a graph                          │
└──────────────────────────────────────────────────────────────────┘
          ▲                    ▲                     ▲
          │                    │                     │
   [Optical GeoTIFF]    [SAR GeoTIFF]        [PNG / JPEG]
```

Each stage has a typed contract, so an engine can be swapped or disabled without breaking the
rest of the run. Orchestration lives in [backend/agent/](backend/agent/) and
[backend/controller/](backend/controller/); job lifecycle in
[backend/job_manager.py](backend/job_manager.py), which runs pipelines on a worker thread so the
API stays responsive while a job is in flight.

| Stage | Module | Responsibility |
|---|---|---|
| **MC1** | [backend/mc1/](backend/mc1/) | Qualify the input: format validation, metadata extraction, modality classification (optical/SAR), spatial and temporal analysis, hard/soft compatibility profile |
| **MC2** | [backend/qwen/](backend/qwen/) | Understand the query and decompose it into a structured `TaskSpec` |
| **MC3** | [backend/mc3_planner/](backend/mc3_planner/) | Tool Registry, workflow planner, dispatcher — decide *which* engines can answer this question given these inputs |
| **MC4a** | [backend/mc4a_vqa/](backend/mc4a_vqa/) | Single-image visual question answering and captioning (PaliGemma) |
| **MC4b** | [backend/mc4b_temporal/](backend/mc4b_temporal/) | Temporal change analysis — learned backbone plus a classical log-ratio / CVA engine |
| **MC4c** | [backend/mc4c/](backend/mc4c/) | Optical–SAR cross-modal fusion (CROMA) with query-conditioned attention |
| **MC5** | [backend/mc5_evidence/](backend/mc5_evidence/) | Normalise every tool's output into a common evidence graph |
| **MC6** | [backend/mc6_verification/](backend/mc6_verification/) | Verify the graph: conflict detection and temporal consistency |
| **MC8** | [backend/mc8_export/](backend/mc8_export/) | Export the result as JSON, GeoJSON, or PDF |

<br>

---

## Try the Demo

**→ [sat-query-ai-mu.vercel.app](https://sat-query-ai-mu.vercel.app/)**

No upload, no signup, no setup. Three sample queries ship with the site, each carrying its own
real imagery, and each one runs the live pipeline when clicked — there are no recorded answers.

### What to do

**1. Open the site and click *Launch Assistant*.**
The home page explains the six stages a query moves through; hover or click any stage to see what
it actually does.

**2. Read the header pill before you start.**
It states the configuration you are about to use — which LLM is running and how many specialist
engines this deployment can execute. This is read live from the backend, not hardcoded.

**3. Click a sample card.** Three are available:

| Card | Question | Imagery |
|---|---|---|
| **New airstrip in the Amazon** | *Has a new airstrip been cleared between these two acquisitions?* | Sentinel-1 SAR pair · S1-AAD |
| **What changed here?** | *What changed between these two images?* | Sentinel-1 SAR pair · S1-AAD |
| **Ask about one image** | *Is there a river in this image?* | Sentinel-2 optical patch · BigEarthNet |

A card whose engine is unavailable on the connected backend is **disabled and shows the backend's
own reason** — that is the honest-capability behaviour working, not a broken button.

**4. Watch the pipeline while it runs.**
The stage rail is driven by the stages the backend actually reports. Nothing is animated on a
timer. On the hosted free-tier backend a change query finishes in a few seconds; with the full
local stack and Qwen3 in the loop it takes roughly 45–90 seconds, and the rail is what shows you
where it is.

**5. Read the answer, then interrogate it:**

- **Confidence badge** — labelled *Model confidence (uncalibrated)*, because calibration is not
  implemented. It is never dressed up as a calibrated probability.
- **Evidence chips** — one per evidence object behind the answer. **Click one**: it highlights
  that region on the map and opens the execution trace at the matching step.
- **Caveats** — assumptions the run made, e.g. *acquisition dates missing; assumed upload order*.
- **Map / Swipe** — switch to *Swipe* and drag the divider to compare before and after.
- **Show execution trace** — every stage, tool call, and verification decision.
- **Exports** — PDF report, GeoJSON regions, JSON trace.

**6. Then try your own.** Attach one or two images (GeoTIFF, PNG, JPEG) and ask. For change
detection, attach the earlier image first — Image A is treated as *before* unless the files carry
acquisition dates.

### Worth trying deliberately

Ask a single-image question while two images are attached. The system returns
`INSUFFICIENT_OBSERVATIONS` and lists the candidates rather than guessing which image you meant.
Refusing to answer is a designed behaviour, and it is documented in
[KNOWN_GAPS.md](KNOWN_GAPS.md).

<br>

---

## Capability Status

The hosted demo runs on a free-tier host with 512 MB of RAM, so the heavier engines are turned
off there. This table is what the system claims for itself — it is also what `GET /api/system`
reports per deployment.

| Engine | Status | Measured |
|---|---|---|
| **Classical Change Detection** (SAR log-ratio / optical CVA) | Working — serves every change query | Pixel F1 **0.104** on 113 S1-AAD pairs ([report](backend/data/reports/classical_cd_s1aad_eval.json)) |
| **Single Image VQA** (PaliGemma 3B) | Working on the full local stack; disabled on the free-tier host | **Not yet measured** on a public VQA benchmark |
| **Temporal Change Analysis** (ChangeMamba) | Unavailable — needs CUDA + `mamba-ssm` | — |
| **Optical–SAR Fusion** (CROMA) | Withheld — loads and runs, but did not pass validation | Matched vs mismatched pairs at chance (0.011 vs 0.011 cosine, 40 pairs) |

Those numbers are not impressive, and they are published anyway. The classical detector localises
*where* a radar signal changed; it does not classify *what* changed, and the answer says so. An
engine that has not earned its place does not get to contribute evidence — CROMA stays disabled
unless `SATQUERY_ENABLE_UNVALIDATED_CROMA=1` is set explicitly.

[KNOWN_GAPS.md](KNOWN_GAPS.md) is the full list, kept current rather than pruned.

<br>

---

## Core Modules

### 1. Input Qualification (MC1)

```
Upload → Format Check → Metadata Extract → Modality Classify → Compatibility Profile
```

Rejects what cannot be analysed before any compute is spent. Modality is classified from
evidence in descending confidence: explicit sensor metadata, then raster properties (band
descriptions `VV`/`VH` → SAR, colour interpretation → optical), then band count and dtype, then
filename convention — and the trace records **which** rule fired and how confident it was.

### 2. Query Intelligence (MC2)

```
Question → Qwen3 Decomposition → Modality Grounding → TaskSpec
```

Qwen3-4B (Q4_K_M via Ollama) decomposes the question into a typed `TaskSpec`. Decomposed
modalities are then grounded against the words actually present in the query, so the planner
cannot invent a modality the user never mentioned. If Qwen3 is unreachable, deterministic Tool
Registry rules take over and the trace marks `planner.query_intelligence = registry_rules`.

### 3. Tool Registry & Planning (MC3)

```
TaskSpec + Input Profile → Capability Match → Observation Binding → Workflow Plan
```

Each tool declares what it needs: minimum and maximum observations, required modalities, whether
the observations must correspond, whether co-registration is required. The planner matches those
requirements against what was actually uploaded, so an impossible request fails with a reason
rather than a bad answer.

### 4. Specialist Engines (MC4)

```
Bound Observations → Engine Execution → Raw Findings
```

Single-image VQA and captioning (PaliGemma), temporal change analysis (learned backbone plus a
classical SAR log-ratio / optical CVA detector), and optical–SAR fusion. Availability is resolved
per deployment and reported honestly — see [Capability Status](#capability-status).

### 5. Evidence Normalisation & Verification (MC5 / MC6)

```
Findings → Evidence Objects → Graph Edges → Conflict Check → Verified Claims
```

Every engine's output becomes an evidence object with a claim, a spatial region, a source model,
and processing parameters. MC6 checks the assembled graph for conflicts and temporal
inconsistency; rejected claims are marked as rejected rather than quietly dropped.

### 6. Answer Synthesis & Export (MC8)

```
Verified Evidence → Answer + Confidence + Caveats → JSON / GeoJSON / PDF
```

Synthesis runs in [backend/job_manager.py](backend/job_manager.py) — Qwen3 writes the answer from
the verified evidence, or [backend/agent/rule_based.py](backend/agent/rule_based.py) composes a
deterministic one when the LLM is unavailable. Either way the answer cites only evidence that
survived verification, confidence is passed through as the raw model value and labelled
uncalibrated, and exports carry the full trace so a reviewer can reconstruct the run without the
UI.

<br>

---

## Technology Stack

### Frontend
```
React 19 • TypeScript • Vite • React Router 7 • React-Leaflet • IBM Plex
```

### Backend
```
Python 3.12 • FastAPI • Uvicorn • Pydantic • Rasterio • Shapely • PyProj • ReportLab
```

### AI / ML
```
Qwen3-4B (Ollama Q4_K_M) • PaliGemma 3B • CROMA • PyTorch • Transformers
```

### Data
```
Sentinel-1 SAR • Sentinel-2 optical • S1-AAD • BigEarthNet • GeoTIFF / WGS84
```

<br>

---

## Getting Started

### Prerequisites

```bash
Python >= 3.12
Node.js >= 18.x
Ollama (optional — without it, the planner uses Tool Registry rules)
```

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
python3 -m uvicorn main:app --port 8000
```

Confirm what this particular server can do before demoing it:

```bash
curl http://localhost:8000/api/system
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev            # http://localhost:5173
```

See [frontend/README.md](frontend/README.md) for routes, configuration, and production builds.

### 3. Optional — Qwen3 for query intelligence

```bash
ollama pull qwen3:4b
```

Without it the pipeline still runs end to end; the trace will simply record that the planner used
registry rules.

### Tests

```bash
cd backend
python3 -m pytest        # 739 tests
```

Tests needing a live API server skip themselves automatically when nothing is listening on
`localhost:8000`.

<br>

---

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/validate` | MC1 validation only — check inputs before committing to a full run |
| `POST /api/query` | Submit a query to the full pipeline; returns a `job_id` immediately. Multipart fields: `files` (1–2 images), `query`, and the optional `query_language` |
| `GET /api/system` | LLM backend and available engines for this deployment |
| `GET /api/jobs/{id}/status` | Job status |
| `GET /api/jobs/{id}/result` | Final answer |
| `GET /api/jobs/{id}/trace` | Execution trace |
| `GET /api/jobs/{id}/evidence_graph` | Evidence graph |
| `GET /api/jobs/{id}/structured_trace` | Structured step-by-step trace |
| `GET /api/jobs/{id}/preview/{observation_id}` | PNG preview of an observation |
| `GET /api/jobs/{id}/export/{format}` | Export as `json`, `geojson`, or `pdf` |

#### `query_language` (optional)

`POST /api/query` accepts `query_language`, the language `query` is written in: `en` (default),
`hi`, `kn`, `te` or `ta`. Anything else is rejected with HTTP 400. Omitting the field behaves
exactly as before, so existing clients are unaffected.

A non-English query is translated to English by the Qwen3 instance already in the pipeline,
**once, before MC2 sees it** (`backend/qwen/translation.py`). Everything downstream — MC2, MC3 and
the specialist engines — reasons over the English string, and the transformation is recorded in
the job trace and the result as `input_translation`, with both the original and the translated
text, so a reviewer can see exactly what the pipeline reasoned over:

```json
"input_translation": {
  "from": "kn", "to": "en", "engine": "qwen3:4b",
  "original": "ಈ ಎರಡು ಚಿತ್ರಗಳ ನಡುವೆ ಏನು ಬದಲಾಯಿತು?",
  "translated": "What changed between these two images?"
}
```

The answer and its caveats are synthesised in English from verified evidence and translated as a
final step; the English originals are kept as `final_answer_en` and `caveats_en`.

**A deployment without the language model cannot serve a non-English query.** The Tool Registry
planner matches English keywords, so running it on another language would answer a question the
pipeline never understood. Such a job terminates as `TRANSLATION_UNAVAILABLE` with no claims and
zero confidence, and says why. English queries on the same deployment are unaffected.

The **execution trace and the exported PDF/GeoJSON/JSON stay English** — they are audit artefacts,
and ReportLab's built-in faces carry no Indic glyphs (KNOWN_GAPS §14).

#### Interface languages and voice input

The interface is available in **English, Hindi, Kannada, Telugu and Tamil**. The switcher is in the
header; the choice is remembered in `localStorage` and the matching script font is fetched only
when that language is selected, so an English visitor downloads no Indic font. Identifiers stay in
Latin script in every language — `MC1`–`MC8`, tool and model names (PaliGemma, ChangeMamba, CROMA,
Qwen3), status enums such as `ABSTAIN`, CRS strings, file formats, API fields and execution-trace
keys — because those are the auditable values.

What is **not** translated, deliberately:

| | Language | Why |
|---|---|---|
| Execution trace | English | It is the audit artefact. When a query was translated, its first section shows the original *and* the English the pipeline reasoned over. |
| PDF / GeoJSON / JSON exports | English | ReportLab's built-in faces carry no Indic glyphs, so an Indic PDF would be empty boxes. The result reports `exports_language: "en"` (KNOWN_GAPS §14). |
| Answer caveats, sometimes | falls back to English | Qwen3-4B does not translate every passage reliably. Measured on one real job: Hindi 2/2, Telugu 2/2, Tamil 1/2, **Kannada 0/2**. A passage it cannot translate is left in English rather than replaced with something plausible and wrong (KNOWN_GAPS §16). |

**Voice input** uses the browser-native Web Speech API. In Chrome that is not on-device — audio is
streamed to Google's speech service and a transcript comes back. Nothing reaches the SatQuery
backend, and no audio is stored, but it is a third party (KNOWN_GAPS §19). Where the API is absent
(Firefox, some Safari builds) the mic is visible but disabled, with the reason in its tooltip.

Recognition is bound to `en-IN`, `hi-IN`, `kn-IN`, `te-IN` and `ta-IN`, one per interface language.
**Which of these the speech service actually accepts has not been confirmed with real speech** —
the Web Speech API publishes no list of supported languages, so the only way to find out is to
speak. Expect Kannada and Telugu to be weaker than Hindi and Tamil. If the service rejects a
language, the mic disables itself for that language only and shows the real reason rather than
failing silently (KNOWN_GAPS §20, §21).

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

<br>

---

## Deployment

[render.yaml](render.yaml) is a Render blueprint for the backend on the free tier (512 MB RAM,
0.1 CPU, sleeps after 15 minutes idle, ephemeral disk). That plan runs Tool Registry rules,
classical change detection, and exports; PaliGemma VQA and CROMA fusion are disabled there
because they need gigabytes of RAM. The frontend deploys to Vercel with `VITE_API_BASE` pointed
at the backend URL.

Running the full-capability stack means running the backend where the weights and Ollama live —
the hosted demo is deliberately the reduced configuration, and the UI says which one you are
talking to.

## Data & Model Weights

Not committed — they are gigabytes, and some carry unresolved dataset licences:

| Asset | How to get it |
|---|---|
| CROMA weights | `backend/download_croma.py` → `backend/mc4c/weights/` |
| BigEarthNet | `backend/download_bigearthnet.py` |
| Feature caches | `backend/scripts/` regenerates `backend/data/features/` |

Evaluation harness and fixtures: [backend/evaluation/](backend/evaluation/).
Adaptation-run results: [backend/data/adaptation/metrics/](backend/data/adaptation/metrics/).

## Known Limitations

[KNOWN_GAPS.md](KNOWN_GAPS.md) records what does not work yet and why — ambiguous target
resolution across two images, omitted evidence-graph edges, uncalibrated confidence, and the
engine gaps in the table above. It is maintained as a live document, not a launch-day artefact.

<br>

---

## Acknowledgments

- **Qwen3** (Alibaba) for the planning model, served through **Ollama**
- **PaliGemma** (Google) for visual question answering
- **CROMA** (Fuller et al.) for the optical–SAR encoder
- **Copernicus / ESA** for Sentinel-1 and Sentinel-2 imagery
- **S1-AAD** Amazon airstrip dataset and **BigEarthNet** for evaluation data
- **OpenStreetMap** contributors for basemap tiles
