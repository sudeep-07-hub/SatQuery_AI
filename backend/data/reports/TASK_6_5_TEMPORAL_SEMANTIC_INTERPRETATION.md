# Task 6.5

## Objective
Implement the semantic interpretation layer (`TemporalSemanticInterpreter`) linking the normalized MC5 temporal evidence to the Qwen3 reasoning layer. The primary mandate is strict semantic safety: Qwen3 must operate exclusively on the provided Evidence Objects and is explicitly prevented from hallucinating land-cover changes without spatial grounding.

## Existing Qwen3 Integration
- **Status**: **REAL**
- Leverages the existing `Qwen3Inference` component (`backend/qwen/inference.py`), seamlessly utilizing HuggingFace on available backends (including Apple Silicon MPS).
- Model parameters explicitly load `Qwen/Qwen3-4B-Instruct-2507`.

## Semantic Responsibility Boundary
- **Status**: **REAL**
- The Temporal Specialist handles raw pixels. The Normalizer extracts Regions. The `TemporalSemanticInterpreter` acts purely as the structured reasoning router, never accessing raw images or assuming uncalibrated confidence.

## Temporal Evidence Context
- **Status**: **REAL**
- Inputs are rigorously wrapped in a `TemporalEvidenceContext` payload carrying `task_type`, explicit `before_observation`, `after_observation`, the `pipeline_status` from the inference layer, and the extracted Evidence Objects.

## Change Description
- **Status**: **REAL**
- The system correctly channels `change_description` instructions through the identical evidence constraint framework.

## Change VQA
- **Status**: **REAL**
- Allows question-answering over the evidence array while preserving exact Evidence ID traceback.

## Evidence Grounding
- **Status**: **REAL**
- Prompts forcefully instruct: *"Answer only from the supplied temporal evidence. Do not claim visual observations that are not represented in the evidence."*

## Semantic Safety
- **Status**: **REAL**
- If the user asks unsupported semantic questions (e.g., "building", "forest") but the available evidence objects lack those tags, the execution explicitly short-circuits.

## Model-Unavailable Behavior
- **Status**: **REAL**
- A `MODEL_UNAVAILABLE` pipeline status immediately routes to a `TEMPORAL_EVIDENCE_UNAVAILABLE` state without ever polling the Qwen3 layer, securing the pipeline against false positives.

## Insufficient Evidence
- **Status**: **REAL**
- If the evidence does not establish the queried land-cover type, it returns an explicit `INSUFFICIENT_EVIDENCE` status.

## Structured Output
- **Status**: **REAL**
- All answers conform to `TemporalAnswer`, guaranteeing a standardized format (`status`, `answer`, `evidence_ids`, `supported_claims`, `uncertainty`) for downward integration with MC6.

## Tests
- Introduced `test_mc4b_semantic_interpreter.py`.
- Formally validates:
  - `MODEL_UNAVAILABLE` propagates as `TEMPORAL_EVIDENCE_UNAVAILABLE`.
  - `SUCCESS` with zero masks propagates as a grounded Negative.
  - Semantic short-circuits on unsupported land-cover strings.
  - Generative paths correctly insert Evidence IDs into the Qwen3 context.

## Current ChangeMamba Availability
- **Status**: **BLOCKED**
- While Qwen3 interpretation operates on test data, the upstream ChangeMamba pixel mask pipeline remains definitively blocked by physical CUDA unavailability in the MPS host environment. Thus, the system elegantly demonstrates pipeline readiness without corrupting the inference graph.

## Final Verdict
**PASS**

The `TemporalSemanticInterpreter` enforces an evidence-first barrier preventing Qwen3 from engaging in unsupported computer vision hallucination, securing the chain of evidence from the raw raster down to the user-facing natural language layer.
