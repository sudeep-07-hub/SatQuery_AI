# Task 6.1

## Objective
Perform a rigorous read-only audit of the existing temporal/change-analysis implementation and establish the formal contract for integrating the Phase 6 REAL temporal specialist (ChangeMamba) securely into the SatQuery AI agent architecture.

## Current Temporal Architecture
The current architecture routes temporal queries through `job_manager.py` (which sets primary_task to `change_detection`) or via Qwen3's advanced workflow planner.
- Inputs pass through MC1 Observation qualification.
- MC3 routes the task to the registered `CHANGE_MAMBA_TOOL`.
- Outputs map into MC5 Evidence Objects, eventually reaching MC6 Verification to validate claims like "increase in built-up area."

## Current Implementation
The `mc4b_temporal/` subsystem currently contains:
- `tool_adapter.py`: Structurally implements the `CHANGE_MAMBA_TOOL` interface.
- `backbone.py`: Contains a `LightweightCNNBackbone`.
- **Classification**: **PARTIAL / STUB**
  The infrastructure (schemas, routing, tool adapters) is rigorously implemented, but the core modeling backbone is a deliberately constructed CNN stand-in intended for CPU-only testing.

## Real vs Mocked Components
- **REAL**: Input qualification, tool registration, precondition validation, evidence object generation, spatial mapping (GeoJSON wrapping), and MC6 conflict resolution.
- **STUB**: The visual backbone (`LightweightCNNBackbone`).
- **PARTIAL**: The semantic extractor / captioner (currently applying heuristic logic over the CNN logits).

## ChangeMamba Availability
**Real ChangeMamba implementation/checkpoint not currently available.**
The environment lacks `mamba-ssm`, CUDA, and the actual ChangeMamba weights. The code explicitly raises a `NotImplementedError` if the true ChangeMamba model is selected.

## MC1 Temporal Input Contract
- **Status**: **REAL**
- The system correctly models acquisition timestamps, modality, sensor, CRS, spatial bounds, overlap, and coregistration constraints within `ObservationProfile`.
- Temporal ordering relies exclusively on `acquisition_date` extraction, maintaining a strict chronological fallback strategy (`bi_temporal`) rather than trusting file indexing.

## Query Routing
- **Status**: **REAL**
- Both the legacy `job_manager.py` (keyword-based) and the advanced Qwen3 planner (`mc3_planner/workflow_planner.py`) accurately route `change_detection` and `bi_temporal` tasks to the temporal specialist.

## Tool Registry
- **Status**: **REAL**
- `CHANGE_MAMBA_TOOL` is correctly registered in `generate_traces.py` and `dispatcher.py`. The contract handles modality filters and preconditions transparently.

## Temporal Output Contract
- **Status**: **REAL**
- The tool adapter produces a hardened contract: `binary_change_mask`, `change_score`, `changed_region_coordinates`, `model_confidence`, `semantics`, and `caption`.

## Evidence Integration
- **Status**: **REAL**
- Output coordinates map deterministically to `claim`, `spatial_region`, and `evidence_type: change_detection_mask` within MC5.1 formats.

## Evidence Graph Integration
- **Status**: **PARTIAL**
- `evidence_graph.py` links `Evidence` nodes to `Claim`, `Model`, `Modality`, and `Region`. However, temporal relationship edge tracking (`compared_with` between observations) is proposed for future implementation, as current nodes loosely bundle timestamps.

## Confidence
- **Status**: **STUB**
- The existing confidence emerges uncalibrated from the lightweight CNN stand-in logic.

## Change Map
- **Status**: **PARTIAL**
- Outputs binary masks mapped geometrically back to coordinates using basic vectorization. Complex multi-class maps remain a future feature.

## Failure Handling
- **Status**: **REAL**
- `tool_adapter.py` rigorously rejects mismatched modalities, unequal CRS, single images, and low coregistration scores, bubbling a `PRECONDITION_FAILED` message directly to MC3 for recovery or abstention.

## Existing Tests
- Comprehensive unit coverage is structurally confirmed for:
  - `test_mc4b.py`: Tool adapter invocation and spatial mapping.
  - `test_mc4c_engine.py` / `test_mc4c_fallback.py`: Planning routing behaviors.
  - `test_mc6_verification.py`: Handles conflicting change claims.

## Dependencies
- Phase 6 remains isolated safely behind the `CHANGE_MAMBA_TOOL` abstraction boundary. Implementing the real model requires modifying `backbone.py` without breaking MC3 planners or MC5 graphs.

## Required Phase 6 Work
The subsequent roadmap for Phase 6 is officially:
1. **6.2 Real Temporal Model Integration**: Procure, install, and instantiate the real ChangeMamba checkpoints (CUDA required).
2. **6.3 Temporal Inference Pipeline**: Connect the real model to the bi-temporal tensor pipeline.
3. **6.4 Temporal Evidence Normalization**: Standardize the ChangeMamba logits into precise MC5 formats.
4. **6.5 Change Description / VQA**: Replace heuristic captioners with VLM-backed (Qwen3) interpretative routing.
5. **6.6 Temporal Verification**: Enhance MC6 conflict handling for ambiguous temporal boundaries.
6. **6.7 Phase 6 Integration + Regression**: System-wide end-to-end tests.

## Limitations
- Execution of real temporal analysis is completely stalled due to the absence of the ChangeMamba model and required CUDA dependencies.

## Final Verdict
**PASS**

The architectural audit is complete, precise boundaries between real plumbing and stub models have been documented, and a formal structural contract has been ratified. The foundation is officially ready for the genuine ChangeMamba integration.
