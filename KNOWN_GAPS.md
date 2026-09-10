# SatQuery AI — Known Architecture & Capability Gaps
*Updated: 2026-09-10*

## 1. Ambiguous Target Resolution (MC2/MC3)
- **Symptom**: When multiple images are uploaded but a single-image query is issued (e.g., "Is there an airstrip in this image?"), the system silently defaults to evaluating `image_1` unless the query explicitly contains words like "second" or "image 2".
- **Impact**: The user is not prompted to disambiguate, and the system does not disclose that it guessed the target image. This violates the "no silent guesswork" agentic principle.
- **Required Fix**: Add a `target_image` field to the MC2 Structured Task Specification schema. If `image_count > 1` and `target_image` cannot be definitively resolved from the query, the planner should halt with `PRECONDITION_FAILED` and prompt the UI to render a disambiguation widget (e.g., "Which image do you mean? [Image 1] [Image 2]"). 
- **Current Status**: Documented as a known gap. The `dispatcher.py` has been temporarily patched to inject an explicit warning into the evidence graph when it falls back to guessing, ensuring the user is at least notified of the assumption.

## 2. Hardcoded Entity Extraction (MC2)
- **Symptom**: The task classifier currently hardcodes `target_entities: ["built-up area"]` and `required_operations: ["temporal_analysis", "spatial_localization"]` regardless of the actual user query.
- **Impact**: The system cannot dynamically extract novel entities (e.g., "airstrip", "deforestation") from natural language, relying instead on generic tool capabilities.
- **Required Fix**: Replace the heuristic `job_manager.py` task parser with a real LLM-based entity extraction and decomposition prompt (the true intended MC2 architecture).
- **Current Status**: Documented as a known gap for the Phase 1 build scope.
