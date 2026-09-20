# SatQuery AI — Known Architecture & Capability Gaps
*Updated: 2026-09-16 (demo prototype stabilization pass)*

## 1. Ambiguous Target Resolution (MC2/MC3) — OPEN
- **Symptom**: With two images uploaded, a single-image question ("Is there a runway?") cannot be bound to one observation.
- **Current behaviour**: The binder returns `AMBIGUOUS` and the job ends as `INSUFFICIENT_OBSERVATIONS` with the candidate images listed. It no longer guesses `image_1`.
- **Required Fix**: Add a `target_image` field to the TaskSpec and a UI disambiguation widget ("Which image do you mean?").

## 2. Hardcoded Entity Extraction (MC2) — RESOLVED
- Query intent now comes from Qwen3 (`qwen3:4b` Q4_K_M via Ollama by default). When Qwen3 is unavailable or cannot resolve the intent, deterministic Tool Registry rules are used and the trace/result say so (`planner.query_intelligence = registry_rules`).

## 3. Omitted Evidence Graph Edges (MC5) — OPEN
- Only `supports`, `derived_from`, and `overlaps` edges exist; `contradicts`, `corroborates`, `precedes` are not implemented.

## 4. Learned Change Detection (MC4B) — OPEN (environment)
- ChangeMamba needs CUDA + `mamba-ssm`; on Apple Silicon it is registered as unavailable.
- Change queries are served by the classical engine (SAR log-ratio / optical CVA). It localises change but does not classify it. Measured on 113 S1-AAD pairs: pixel F1 0.10, precision 0.06, recall 0.28 (`backend/data/reports/classical_cd_s1aad_eval.json`).

## 5. CROMA Optical+SAR Fusion (MC4C) — WITHHELD (unvalidated)
- CROMA loads and runs on real Sentinel-2/Sentinel-1 patches, but its optical/SAR agreement score does not separate matching from non-matching BigEarthNet pairs (40 pairs: mean cosine 0.011 vs 0.011; retrieval at chance). The tool is disabled in real mode unless `SATQUERY_ENABLE_UNVALIDATED_CROMA=1`.
- The BigEarthNet-v2 land-cover head scored micro-F1 0.28 on 80 test patches under every band ordering tried, so its tags are not used as evidence.

## 6. Confidence Calibration (MC6.2) — OPEN
- All confidences are raw/uncalibrated and labelled as such (PaliGemma token probability, Otsu separability, change strength).

## 7. Single-Image VQA Quality (MC4A) — OPEN
- `google/paligemma-3b-pt-224` is the pretrained (not mix/fine-tuned) checkpoint; answers are short and unreliable on SAR (confidence is down-weighted by 0.3 for SAR inputs).

## 8. Horizontal Overflow on Narrow Screens (Frontend) — OPEN
- **Symptom**: `/assistant` scrolls horizontally on a phone. `document.scrollWidth` measures 401 px at both a 320 px and a 375 px viewport.
- **Cause**: the chat header's `ModelSelector` is a native `<select>` sized by its widest option ("More controllers — coming soon (not available)"): 329 px inside a 400 px label that does not shrink. The composer row itself fits at 320 px.
- **Not caused by the composer work** — found while verifying composer alignment at 320 px (FIX_LOG.md, 2026-09-20) and left alone, as the header is outside that change's scope.
- **Required fix**: let the control shrink (`min-width: 0` on the label, a `max-width`/truncated trigger), or replace the native `<select>` — it is a placeholder with exactly one real option today.
