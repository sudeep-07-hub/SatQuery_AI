# Task 6.4

## Objective
Implement the temporal evidence normalization layer to convert validated `RawTemporalOutput` structures into spatially accurate MC5 `Evidence` objects. The layer strictly preserves the boundary between raw geometric model outputs and higher-level semantic claims while gracefully handling the blocked model execution state.

## Existing Evidence Architecture
- The previous evidence normalizer blindly converted the legacy fake CNN outputs into MC5 Evidence Objects, fabricating semantic interpretations such as "Change detected. Changed area... % of scene." without strict traceability or proper failure handling.

## Raw Temporal Output
- The `normalize_to_evidence` function was updated to consume the `TemporalResult` pipeline wrapper instead of an unstructured dictionary.

## Output Validation
- Validation is firmly enforced: If `temporal_result.status != "SUCCESS"`, the system returns `[]` (zero evidence objects). The prior version attempted to process empty dictionaries.

## Activation and Thresholding
- **Status**: **REAL**
- Since the upstream pipeline defines a `binary_mask` (conceptually after an activation and a threshold of 0.5), the normalizer uses the populated mask without needing to apply a redundant second threshold.

## Mask Normalization
- **Status**: **REAL**
- The system extracts active pixels (`val > 0`) from the mask array.

## Spatial Transformation
- **Status**: **REAL**
- The `_pixel_to_geo` function correctly applies the `affine_transform` `[a, b, c0, d, e, f0]` from the MC1 metadata to translate pixel coordinate centers `(c + 0.5, r + 0.5)` back into precise CRS polygon bounds.

## Region Extraction
- **Status**: **REAL**
- Bounding geometries are safely derived from the transformed active pixels in standard GeoJSON `Polygon` format.

## Evidence Object Mapping
- **Status**: **REAL**
- Formally maps into the required schema: `evidence_id`, `evidence_type` (`bitemporal_change_detection`), `claim`, `spatial_region`, `modality` (`optical`), `timestamp` (`{"t1", "t2"}`), `source_model`, `source_input`, and `processing_parameters`.

## Temporal Provenance
- **Status**: **REAL**
- Preserves explicit `t1` and `t2` keys mapped to the exact chronological `before_observation_id` and `after_observation_id`.

## Evidence Graph Integration
- **Status**: **REAL**
- The Evidence Object outputs match the MC5.1 schema exactly, guaranteeing safe integration into the graph.

## Failure Handling
- **Status**: **REAL**
- Returns 0 Evidence Objects silently and gracefully if the model is unavailable, blocking pollution of the graph.

## Semantic Safety
- **Status**: **REAL**
- The previous hardcoded synthesized claims were completely stripped. The new claim is purely observational: *"Model-predicted change region detected between {obs_1} and {obs_2}."*

## Tests
- Added `test_mc4b_evidence_normalizer.py`.
- Includes `test_model_unavailable_returns_no_evidence`.
- Includes `test_pixel_to_geo_transformation` confirming affine geometry.
- Includes `test_semantic_safety_and_provenance`.

## Final Verdict
**PASS**

The normalizer firmly protects the MC5 Evidence Graph by establishing a strictly spatial, non-hallucinated temporal mapping layer that explicitly respects the `MODEL_UNAVAILABLE` blocked boundary from Task 6.2/6.3.
