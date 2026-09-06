---
name: SatQuery AI — Project Brief
description: >
  Master project brief for SatQuery AI, a query-driven, evidence-grounded agentic
  remote-sensing assistant. Covers system overview, all 8 pipeline stages, current
  build scope (Stage 1 only), and tech stack. Reference this skill before starting
  any task in this workspace.
---

# SatQuery AI — Project Brief

## System Overview

SatQuery AI lets a user upload 1–2 remote-sensing images (optical and/or SAR — GeoTIFF,
TIFF, PNG, JPEG from sensors like Sentinel-1/2, Landsat-8/9, Cartosat, RISAT) plus a
natural-language query (e.g. "Has built-up area increased?"), and returns an
**evidence-grounded, confidence-calibrated answer** with visual proof (change maps,
bounding boxes, masks) rather than a free-text hallucination.

## Full Pipeline (8 Stages)

| Stage | Name | Purpose |
|-------|------|---------|
| 1 | **Geo-Input Qualification & Preprocessing (MC1)** | Classify uploads, detect modality/sensor, check spatial & temporal compatibility, produce a Structured Input Profile JSON |
| 2 | Query Decomposition | Break the NL query into sub-tasks |
| 3 | Tool Registry | Match sub-tasks to specialist models/tools |
| 4 | Specialist Models | Run inference (change detection, segmentation, etc.) |
| 5 | Evidence Graph | Assemble outputs into a structured evidence graph |
| 6 | Verification Loop | Cross-check evidence consistency & confidence |
| 7 | Answer Synthesis | Generate a grounded, cited answer from the evidence |
| 8 | Audit GUI | Interactive front-end for reviewing evidence + answer |

## Current Build Scope

**Stage 1 ONLY** — Geo-Input Qualification & Preprocessing Layer ("MC1").

- **Input:** user-uploaded image(s) + natural-language query.
- **Output:** a "Structured Input Profile" JSON object.
- **Determines automatically:**
  - What the user uploaded (format, count).
  - What modality/sensor each image is.
  - Whether the images are spatially compatible (CRS, footprint overlap).
  - Whether they are temporally related (co-registration, acquisition-date relationship).
  - Whether the requested task can actually be executed given the above.

> **Do NOT implement Stages 2–8.** A later phase will replace any stubs left for those.

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| Backend | Python 3.11, FastAPI, GDAL + rasterio (raster/CRS handling), Pillow (plain PNG/JPEG) |
| Frontend | React + TypeScript, Vite, plain `fetch` (no heavy state library) |
| Repo layout | Monorepo — `/backend` and `/frontend` as separate folders |

## Key Design Principles

1. **Evidence over hallucination** — every answer must cite visual/spatial evidence.
2. **Confidence calibration** — outputs carry calibrated confidence scores.
3. **Modality awareness** — the system must distinguish optical vs. SAR and handle each appropriately.
4. **Spatial rigour** — CRS alignment and footprint overlap are first-class concerns.
5. **Phased build** — each stage is self-contained; later stages replace stubs, not refactor core logic.
