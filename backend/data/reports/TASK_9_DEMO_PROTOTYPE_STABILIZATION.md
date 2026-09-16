# Task 9 — Demo Prototype Stabilization

*2026-09-16 · Supersedes `FINAL_TEST_REPORT.md`, `COMPONENT_MAP.md`, and `TASK_7_1_FULL_AGENTIC_INTEGRATION.md`*

## 1. Verdict

**Working prototype on a 16 GB Apple Silicon Mac.** A user can upload images in the UI and ask a question. The pipeline then runs MC1 → Qwen3 → Tool Registry → a specialist engine → evidence graph → verification → Qwen3 answer → exports, on the pixels the user actually uploaded.

- Verified end to end over HTTP and in a headless-Chrome render of the UI.
- Test suite: 694 passed, 2 failed, 31 skipped. The 2 failures were stale assertions, since updated (§6).

## 2. How to run the demo

```bash
# 1. LLM (quantized Qwen3, ~2.5 GB)
ollama serve            # already running as the Ollama app on this machine
ollama pull qwen3:4b    # already pulled

# 2. Backend
cd backend && python3 -m uvicorn main:app --port 8000

# 3. Frontend
cd frontend && npx vite --port 5173
```

Environment options:

| Variable | Default | Meaning |
|---|---|---|
| `SATQUERY_QWEN_BACKEND` | `ollama` | `ollama` (Q4_K_M), `transformers` (bf16, ~8 GB RAM), or `none` |
| `SATQUERY_PLANNER_FALLBACK` | `registry` | `registry`: use Tool Registry rules if Qwen3 is down. `strict`: stop with MODEL_UNAVAILABLE |
| `SATQUERY_OLLAMA_MODEL` / `SATQUERY_OLLAMA_URL` | `qwen3:4b` / `http://localhost:11434` | |
| `SATQUERY_ENABLE_UNVALIDATED_CROMA` | unset | Set to `1` to expose the unvalidated CROMA fusion tool |

**Recommended demo inputs:** `S1-AAD .../Change_detection/before/_ID_32.tif`, then `after/_ID_32.tif` (upload order = before, after). Query: *"What changed between these two images?"*

- The detected change lies along the new airstrip in the ground-truth mask.
- A shareable result link has the form `http://localhost:5173/?job=<job_id>`.

## 3. Qwen3: quantized model plus Tool Registry fallback

| Path | When | Latency |
|---|---|---|
| Ollama `qwen3:4b` Q4_K_M (default) | Ollama reachable | ~11 s per LLM call; ~60–115 s per change query (4 calls) |
| Tool Registry rules | Ollama down, Qwen3 output unusable, or `SATQUERY_QWEN_BACKEND=none` | < 1 s |

Qwen3 thinking mode is disabled for the structured JSON calls (`think: false`).

When the fallback triggers, deterministic rules take over, and every result records which planner produced each step in `result.planner`:

- **Intent:** keyword rules plus the shape of the inputs (image count and modality).
- **Tool choice:** the best enabled registry capability for the task.
- **Recovery:** another enabled capability that supports the same task.
- **Answer:** an evidence-only template.

The Tool Registry is now **availability-aware per job**. Each engine reports whether it can run here, unavailable engines are disabled before selection, and the trace lists every engine with its status and reason.

## 4. Issues fixed

| # | Issue | Fix |
|---|---|---|
| 1 | Specialists received `torch.rand` noise instead of the upload | `raster_io.py` saves each upload per job and loads the real pixels. Engines read the observation bound to them. |
| 2 | MC1 crashed on every georeferenced GeoTIFF (shapely polygon passed where a dict is required) | Footprint converted to GeoJSON; affine transform normalised to 6 values |
| 3 | CROMA real mode silently fell back to zero tensors | Fallback removed from real mode. Missing inputs now fail with explicit reasons. |
| 4 | Missing CRS was reported as overlap/co-registration 1.0 | Now `None`. Same-size plain images are flagged `assumed_pixel_aligned` with a warning. CRS mismatches are reprojected before overlap is computed. |
| 5 | Hard-coded 0.85 confidence on invented "Generated result from …" evidence | Removed. Completed tools with no evidence add a caveat. PaliGemma evidence (previously dropped) is now kept. |
| 6 | `smoke_test` in any query bypassed MC1 | Honoured only in `execution_mode="fixture"` |
| 7 | Evaluation runner rewrote mock answers to match ground truth | Removed; predictions are scored as produced |
| 8 | `mc3_planner/dispatcher.py` called `normalize_to_evidence` with the old signature | Non-success results now propagate; no invented evidence |
| 9 | Change detection impossible on this Mac (ChangeMamba needs CUDA) | New `classical_change_detection` engine: SAR log-ratio or optical CVA, georeferenced regions, change statistics, mask PNG |
| 10 | PaliGemma specialist invented a 100-pixel footprint polygon | Uses the real WGS84 footprint, or none |
| 11 | PaliGemma reloaded per job (memory) | One shared, lazily loaded instance per process |
| 12 | Before/after binding failed whenever acquisition dates were missing (all S1-AAD files) | Upload order is used and reported as an explicit caveat |
| 13 | Verification failed the whole job on any trigger | Per-evidence verification: low-confidence items are rejected and shown struck through; global triggers become caveats |
| 14 | API/UI never finished for `MODEL_UNAVAILABLE` / `INSUFFICIENT_OBSERVATIONS` | Shared terminal-status list in the API and UI |
| 15 | UI: hooks-order bug in ResultPanel; map fed UTM metres as lat/lon; `json_trace` export 404; `[object Object]` in trace | Rewritten ResultPanel/map. WGS84 overlays of the uploaded image, change mask and regions. Plain-image viewer for non-georeferenced input. Export alias added. |
| 16 | Tests: stale API names, `backend.` imports, async tests without a plugin, live-server tests, `/tmp` data | `backend/conftest.py` fixes imports and async tests and skips live tests without a server. Stale tests updated (§6). |

New endpoints:

- `GET /api/system`: LLM backend and engine availability.
- `GET /api/jobs/{id}/preview/{image_n}`: PNG rendering of an uploaded GeoTIFF.

## 5. Measured capability (no numbers invented)

| Capability | Measurement | Result |
|---|---|---|
| Classical SAR change detection | 113 S1-AAD pairs with ground-truth masks (`classical_cd_s1aad_eval.json`) | Pixel precision 0.064, recall 0.282, F1 0.104, IoU 0.055 |
| Same, tuned on half of the pairs | Held-out other half | F1 0.127 (not adopted; gain too small to justify the tuning) |
| CROMA optical↔SAR agreement | 40 matched vs shuffled BigEarthNet pairs, two preprocessing variants | Matched 0.011 vs unmatched 0.011 cosine; retrieval top-1 0.03–0.10 (chance 0.025) → **withheld** |
| BigEarthNet-v2 land-cover head | 80 BigEarthNet test patches, 12 band/label orderings | Best micro-F1 0.28 → tags not used as evidence |
| Qwen3 (Ollama) intent classification | Spot checks | "Has a new airstrip been cleared…" → `change_vqa`; "Is there a runway…" → `single_image_vqa` |

What the classical detector can and cannot claim:

- It finds where the radar or optical signal changed, including the airstrip in `_ID_32`.
- It also produces many false alarms from speckle.
- It cannot say what changed. The answer prompt states this, and Qwen3's answers repeat it (e.g. "cannot confirm if a new airstrip has been cleared").

## 6. Test results

| Run | Result |
|---|---|
| Before this work | 611 passed, 79 failed, 1 skipped, 7 files not collectable |
| After, full suite with live API server + Ollama | 694 passed, 2 failed, 31 skipped (13 min) |
| The 2 failures | Asserted the removed invented footprint polygon; updated to expect `None` (or the real footprint when provided) |
| Fixture demo scenarios D1–D4 (`test_demo_scenarios.py`) | 4/4 pass. D3 now expects `INSUFFICIENT_EVIDENCE`, because the only fusion engine fails and nothing can replace it. |

Test changes that alter expectations, not just fix plumbing:

- **Registry now has 4 tools:** `classical_change_detection` added.
- **Geospatial output CRS:** EPSG:4326, as RFC 7946 GeoJSON requires.
- **Temporal relationship name:** `bi_temporal`, per the FIX-1.2 contract.
- **Task 5.8R readiness:** the missing-shards "blocker" was resolved (shards now cached), so the test checks shard presence without loading 8 GB.
- **Live API tests:** rewritten for the current contract: SAR+SAR change detection runs; `smoke_test` over the API does not bypass MC1.

Skipped tests and why:

- **ChangeMamba success-path tests (22):** skipped when CUDA/mamba-ssm is absent.
- **Tests reading `/tmp/ben_test` (7):** that directory was cleared.
- **One superseded normaliser test.**

## 6b. Follow-up fix (2026-09-17): single-image VQA always stopped

| Symptom | Root cause | Fix |
|---|---|---|
| Every single-image question (`.tif` or `.png`) ended as `INSUFFICIENT_OBSERVATIONS: Missing required modalities: {'optical'}` | Qwen3's decomposition step added `required_modalities=["optical"]` to generic questions ("Describe this image."). S1-AAD `.tif` files are SAR and S1-AAD `.png` files have unknown modality, so binding refused them. | `qwen/pipeline.py` keeps an optical/SAR requirement only if the query names it as a whole word ("SAR", "radar", "Sentinel-2", "Landsat", ...) |
| "Describe this image." returned one-word answers such as "close" | Captioning was sent as a question (`answer en Describe this image.`); the task type never reached PaliGemma | The job manager passes the subtask's `primary_task` to the engine; PaliGemma uses `caption en` for captioning |
| 10-band Sentinel-2 GeoTIFFs were shown to PaliGemma in grayscale | The RGB band selection covered only 3–4 band and ≥12 band stacks | 5–11 band stacks use B04/B03/B02 |
| SAR answers looked authoritative | The domain-mismatch flag was internal only | Visible caveat naming the file and its detected modality. When every answer is rejected, the message says so instead of "no evidence was generated". |

Verified through the running server:

| Input | Question | Result |
|---|---|---|
| Sentinel-2 patch (BigEarthNet labels include broad-leaved forest and inland waters) | "Describe this image." | "satellite image of the forest" (0.41) |
| Same patch | "Is there a river in this image?" | "yes" (0.89) |
| S1-AAD `.tif` / `.png` | Single-image questions | Now reach PaliGemma. Answers carry the SAR warning and are often rejected for low confidence (e.g. caption "the sun as seen by the spacecraft", 0.11). |

## 7. Open items

See `KNOWN_GAPS.md`. The ones most visible in a demo:

1. **Two images + a single-image question:** returns `INSUFFICIENT_OBSERVATIONS` (asks for disambiguation) instead of picking an image.
2. **PaliGemma load time:** ~2 min on first use, with disk offload under memory pressure. The `pt-224` checkpoint gives short answers and is unreliable on SAR.
3. **Change detection:** location only (no semantics); precision is low.
4. **Qwen3 4B answer details:** can occasionally misstate a detail from the evidence list (e.g. region counts). The evidence cards are authoritative.
5. **Background map tiles:** OpenStreetMap needs internet; uploaded imagery and overlays render without it.
