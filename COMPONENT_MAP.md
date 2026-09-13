# SatQuery AI — Component Map (Phase 0 Reconnaissance)
*Generated: 2026-09-12 | Phase 0 — READ-ONLY, no fixes applied*

## Component Mapping

| MC | Contract Name | File(s)/Module(s) | Status |
|---|---|---|---|
| MC1 | Geo-Input Qualification & Preprocessing | `mc1/pipeline.py`, `mc1/format_validator.py`, `mc1/metadata_extractor.py`, `mc1/classifier.py`, `mc1/spatial_analyzer.py`, `mc1/temporal_analyzer.py` | **BUILT** |
| MC2 | Query Understanding & Task Decomposition | `job_manager.py` L111–L138 (inline heuristic parser, not a real module) | **PARTIAL** — hardcoded entity extraction, keyword matching instead of LLM decomposition |
| MC3.1 | Tool Registry | `mc3_planner/tool_registry.py` | **BUILT** |
| MC3.2 | Workflow Planner | `mc3_planner/workflow_planner.py` | **BUILT** |
| MC3 (dispatch) | Plan Executor | `mc3_planner/dispatcher.py` | **BUILT** |
| MC4A | Single-Image Perception (PaliGemma-3B) | `mc4a_vqa/specialist.py`, `mc4a_vqa/paligemma_adapter.py`, `mc4a_vqa/mock_adapter.py`, `mc4a_vqa/tool_entry.py` | **BUILT (FORCE-MOCKED)** — Real PaliGemma adapter exists (`paligemma_adapter.py` L24 uses `"answer en"` prefix ✓) but `specialist.py` L25-26 hard-raises to force mock. Mock returns constant confidence 0.95 and canned/echo answers. |
| MC4B | Temporal Engine (ChangeMamba) | `mc4b_temporal/tool_adapter.py`, `mc4b_temporal/backbone.py`, `mc4b_temporal/change_head.py`, `mc4b_temporal/localization.py`, `mc4b_temporal/semantics.py`, `mc4b_temporal/captioner.py`, `mc4b_temporal/config.py` | **BUILT (FALLBACK BACKBONE)** — Config L8: `BACKBONE = "lightweight_cnn"`. Real ChangeMamba backbone not implemented (L117-120 raises `NotImplementedError`). The `LightweightCNNBackbone` is a CPU stand-in with structurally-compatible output but no pretrained weights — produces random-ish outputs. |
| MC4C | Cross-Model Engine (Optical+SAR) | `mc4c/engine.py`, `mc4c/optical_encoder.py`, `mc4c/sar_encoder.py`, `mc4c/fusion.py`, `mc4c/verification.py`, `mc4c/semantic_head.py`, `mc4c/schema.py`, `mc4c/croma.py` | **BUILT** — Full CROMA-based pipeline with Pydantic schema. Not wired into `job_manager.py` dispatch path. Not registered in Tool Registry. Requires both optical+SAR input and CROMA weights. |
| MC5.1 | Evidence Normalizer | `mc4b_temporal/evidence_normalizer.py`, `mc4a_vqa/evidence_normalizer.py` | **BUILT** — Recently rewritten (MC5 protocol). 10-field contract-compliant objects. |
| MC5.2 | Spatial Evidence Graph | `mc5_evidence/evidence_graph.py`, `mc5_evidence/schemas.py` | **BUILT** — 7 node types, 3 edge types (`supports`, `derived_from`, `overlaps`). Missing: `contradicts`, `corroborates`, `precedes`. |
| MC6.1 | Evidence Verifier | `mc6_verification/verifier.py` | **PARTIAL** — Keyword-based conflict detection (checks claims for "increased"/"no change" contradictions). NOT a cross-modal geometric verifier. Uses raw model confidence thresholds only. |
| MC6.2 | Confidence Calibrator | (none) | **NOT BUILT** — `verifier.py` uses raw model confidence compared to `LOW_CONFIDENCE_THRESHOLD`, not calibrated confidence. |
| MC6.3 | Reliability/Abstention Decision | `mc6_verification/verifier.py` (RE_PLAN/INSUFFICIENT_EVIDENCE logic) | **PARTIAL** — The verifier outputs VERIFIED/RE_PLAN_REQUIRED/INSUFFICIENT_EVIDENCE but the re-plan loop in `job_manager.py` L205-212 just calls `verify()` twice and then forces INSUFFICIENT_EVIDENCE. Not a real re-plan. |
| MC7.1 | Evidence-Constrained Answer Generator | `job_manager.py` L225-255 (inline answer assembly) | **PARTIAL** — Ad-hoc: for VQA it echoes `textual_answer`; for change detection it joins claims with "Based on the evidence,". Not evidence-constrained per contract (no MC6 output consumed to gate the answer). |
| MC7.2 | Geospatial Visualization & Reporting | `frontend/src/components/results/BeforeAfterViewer.tsx`, `frontend/src/components/results/ResultPanel.tsx` | **BUILT** — Leaflet map with GeoJSON overlay, claim↔region click highlighting. |
| MC8 | Audit / Reporting / GUI export | `mc8_export/exporter.py`, `main.py` (API routes) | **BUILT** — JSON trace (with `structured_trace`), GeoJSON, PDF, PNG heatmap. |

## Deviations Found (contract vs. implementation)

### DEV-1: MC4A is force-mocked
`specialist.py` L25-26 has `raise Exception("Forced mock for integration tests")`, meaning PaliGemma never actually runs. The mock adapter returns constant confidence `0.95` and canned text. **This means every MC4A "test" in the system produces fabricated outputs.** Per protocol §0 rule 1, any test against the mock is NOT a real test.

### DEV-2: MC4B backbone is a lightweight CNN stand-in, not ChangeMamba
`config.py` L8: `BACKBONE = "lightweight_cnn"`. The `get_backbone("changemamba")` path raises `NotImplementedError`. The LightweightCNN produces structurally-valid but untrained/random outputs. The system honestly labels this as `source_model: "CHANGE_MAMBA_TOOL"` in evidence objects, which is misleading — it's not ChangeMamba running, it's a random CNN.

### DEV-3: MC2 is a hardcoded heuristic, not real query understanding
`job_manager.py` L115-138 uses keyword matching (`"changed"`, `"increase"`, `"decrease"`) to set `temporal_requirement` and hardcodes `target_entities: ["built-up area"]`. This is documented in `KNOWN_GAPS.md` §2.

### DEV-4: MC6 verification is keyword-based, not geometric/model-aware
`verifier.py` checks for string patterns like `"increased"` and `"no change"` in claims. It does not do mask-level conflict detection, spatial overlap verification, or confidence calibration.

### DEV-5: MC7 answer assembly is ad-hoc
`job_manager.py` L242-245: VQA path just echoes `textual_answer`, change detection path joins claims with string concatenation. No evidence-constraint logic.

### DEV-6: No `generate_dynamic_json`, no `STANet`, no `BIGEARTHNET_TOOL`
The protocol references these from a prior codebase iteration. They do not exist in the current code. `STANet` fallback is not implemented — there is no alternative change detection model.

### DEV-7: `CHANGE_MAMBA_TOOL` registry entry declares `supported_modalities: ["optical", "sar"]`
This was previously fixed (FIX-2.1, FIX-4.2). The registry now accepts both optical+optical and sar+sar. The `same_modality` constraint validation was added in Phase 0 of the MC5 protocol. **However:** the underlying backbone is a random CNN (DEV-2), so "supporting SAR" is structurally true but semantically meaningless for real inference.

### DEV-8: PaliGemma prompt prefix is correct
`paligemma_adapter.py` L24: `prompt = f"<image>answer en {query}"`. The `"answer en"` prefix is correct per the Phase 3 regression check. **However**, this code never runs because of DEV-1.

### DEV-9: MC4C exists but is not wired into the dispatch path
`mc4c/engine.py` has a complete CROMA-based cross-modal engine with Pydantic schema validation, but `dispatcher.py` L20-28 only knows about `CHANGE_MAMBA` and `PALIGEMMA_VQA`. MC4C is not registered in the Tool Registry and not callable from `job_manager.py`.

### DEV-10: `result.confidence` in export is raw model confidence, not calibrated
`job_manager.py` L251: `"confidence": max([ev["confidence"] for ev in evidence])`. This is the raw model confidence (from mock: always 0.95 for VQA, or from untrained CNN for MC4B). The PDF report (`exporter.py` L106) labels this as `"Confidence Score"` without qualifying it as raw/uncalibrated. The UI (`ResultPanel.tsx` L50) shows it with High/Medium/Low labels.

## Test Data Inventory

### Confirmed present and readable:
- **S1-AAD SAR GeoTIFFs**: 2,419 files at `/Users/sukesh/Desktop/satquery/S1-AAD Sentinel-1 Amazon Airstrip Dataset/S1-AAD Sentinel-1 Amazon Airstrip Dataset/Images_geotiff/`
  - Sample: `_ID_33.tif` — 201×200px, 1 band, float64, CRS EPSG:32722, 10m GSD
  - These are **single-date** chips (no built-in temporal pairs — bi-temporal pairs must be constructed from two different `_ID_*.tif` files of the same airstrip)

### NOT present:
- No optical GeoTIFFs (Sentinel-2 or Copernicus) — MC4C cross-modal testing cannot use real data
- No OSCD or Ebel-et-al samples
- PNG test images at `frontend/public/` (`test_sar.png`, `test_optical.png`) are not georeferenced GeoTIFFs

### Phase 0 Gate Assessment:
- ✅ At least one real single SAR image confirmed readable (`_ID_33.tif`)
- ✅ A bi-temporal SAR pair can be constructed (e.g., `_ID_33.tif` + `_ID_27.tif` — different airstrips but same CRS)
- ⚠️ No real optical imagery available for MC4C or cross-modal tests
