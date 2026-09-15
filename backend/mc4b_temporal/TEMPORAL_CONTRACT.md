# Temporal Specialist Architecture Contract (MC4B)

## 1. Purpose
The temporal specialist is responsible for identifying, describing, and localizing changes between two strictly co-registered observations of the same geographic area acquired at different timestamps. It acts as the definitive change-analysis node in the MC4 capability set.

## 2. Supported Task Types
The real temporal specialist must officially support:
- `binary_change_detection`: High-confidence spatial masking of structural changes.
- `semantic_change_detection`: Multi-class classification of detected changes (e.g., vegetation loss, urban expansion).
- `change_captioning` / `change_description`: Natural language description of changes.
- `change_localization`: Extraction of changed regions as geospatial coordinates.
- **[PROPOSED]** `change_vqa`: Targeted answering of temporal queries (e.g., "Did a road appear here?").

## 3. Input Contract
- **Image Count**: Exactly 2 observations (`image_1`, `image_2`).
- **Modality**: Both observations must be of the same modality (optical + optical OR SAR + SAR).
- **Coregistration**: Pre-verified high-confidence spatial alignment.

## 4. Observation Contract (T1/T2)
The conceptual mapping requires definitive chronological sorting:
- **Observation A (T1)**: The chronologically earlier acquisition.
- **Observation B (T2)**: The chronologically later acquisition.

## 5. Temporal Ordering
- Temporal ordering MUST be derived from `ObservationProfile.temporal.timestamp` / `acquisition_date`.
- The system must NEVER infer order blindly from filenames, list indexing, or archival order.
- **Fallback Behavior**: If specific dates are missing but both inputs are successfully loaded and identical in modality, the system defaults to a generic `"bi_temporal"` relationship. The model must process them positionally but abstains from absolute chronological descriptions unless order is explicitly injected.

## 6. Validation Requirements
MC1 and MC4B strictly enforce the following rules:
- **HARD REQUIREMENT**: Exactly 2 observations.
- **HARD REQUIREMENT**: Compatible spatial coverage (`min_spatial_overlap`).
- **HARD REQUIREMENT**: Valid CRS compatibility.
- **HARD REQUIREMENT**: Same modality.
- **SOFT REQUIREMENT**: Valid deterministic timestamps. (Missing timestamps drop context to `bi_temporal` but allow execution).

## 7. Model Contract
- **Model**: A real visual state-space model (ChangeMamba) capable of modeling long-range spatial context across temporal pairs.
- **Architecture**: Siamese encoder-decoder producing multi-scale representations.
- **Inference Mode**: Inference-only, no in-context fine-tuning.

## 8. Output Contract
The temporal specialist must output a standardized schema:
- **REQUIRED**: `binary_change_mask` (0 or 1 per pixel).
- **REQUIRED**: `change_score` (fraction of changed pixels).
- **REQUIRED**: `changed_region_coordinates` (GeoJSON geometries).
- **REQUIRED**: `model_confidence` (calibrated or uncalibrated float).
- **OPTIONAL**: `semantics` (semantic distribution).
- **OPTIONAL**: `caption` (linguistic interpretation of change).
- **FUTURE**: `change_vqa_answer` (targeted textual response).

## 9. Evidence Contract
Outputs naturally map to the MC5.1 Evidence Object:
- `claim`: Sourced from the generated `caption` or structural templates.
- `evidence_type`: `change_detection_mask` / `semantic_change_mask`.
- `spatial_region`: The bounding box or polygon of the active changed area.
- `modality`: The underlying matched sensor modality.
- `source_model`: `CHANGE_MAMBA` (or strictly logged fallback).

## 10. Evidence Graph Integration
The MC5 Evidence Graph must map the temporal relationship accurately.
- `Evidence` node supports -> `Claim` node.
- `Evidence` node derived_from -> `Model` node.
- `Evidence` node overlaps -> `Region` node.
- **[PROPOSED]** New edge: `Observation(T1)` -> `compared_with` -> `Observation(T2)`. The evidence object should be formally tied to the differential graph edge.

## 11. Confidence Contract
- **Source**: `model_confidence` derived directly from softmax probabilities or logits distribution.
- **Current Status**: Uncalibrated model score. It must NOT be substituted with a hard-coded or random placeholder in production.

## 12. Change-Map Contract
- **Current**: Binary mask tensor evaluated on a threshold (`> 0.5`).
- **Desired Future Form**: Georeferenced raster patches integrated with polygon boundaries mapping directly to the client UI.

## 13. Failure Contract
Expected failure categories to strictly handle:
- `insufficient_observations` (**HARD FAILURE**)
- `spatial_incompatibility` (**HARD FAILURE**)
- `unsupported_modality` (**HARD FAILURE**)
- `model_unavailable` (**HARD FAILURE**)
- `temporal_order_missing` (**RECOVERABLE** -> execute as unordered difference)
- `cloud_occlusion_excessive` (**REPLAN TRIGGER**)

## 14. Provenance Requirements
- The `source_model` field must carry the exact executing architecture and model revision.
- If a fallback proxy (e.g., LightweightCNN) executes, it MUST broadcast its placeholder status explicitly.

## 15. Determinism Requirements
- The model inference must be 100% deterministic given identical tensor inputs.
- Stochastic masks, randomly generated noise placeholders, or random heuristics are formally classified as defects.

## 16. Future Implementation Requirements
- **Phase 6.2**: Establish real ChangeMamba model weights and CUDA dependencies.
- **Phase 6.3**: Connect the bi-temporal tensor path directly to ChangeMamba inference.
- **Phase 6.4 - 6.7**: Normalization, VQA integration, and MC6 Verification resolution.
