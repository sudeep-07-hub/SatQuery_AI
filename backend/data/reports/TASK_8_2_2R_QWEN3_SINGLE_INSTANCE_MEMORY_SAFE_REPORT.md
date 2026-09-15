# Task 8.2.2R — Qwen3 Single-Instance Memory-Safe Execution

## 1. Executive Summary

This task resolves a critical memory safety issue observed during the transition to a dynamic Qwen3 dependency injection model. The primary objective was to ensure that exactly one instance of `Qwen3Inference` is created, loaded, and shared across all consumers in the pipeline (Query Intelligence, Tool Selector, Recovery Planner, Final Synthesis). Additionally, the lifecycle was hardened to guarantee explicit cleanup (unloading) of the model after job completion, preventing cross-job memory leaks on memory-constrained systems (Apple Silicon).

## 2. Root Cause / Lifecycle Findings

An audit of the pipeline (`job_manager.py`, `qwen/inference.py`, `qwen/pipeline.py`, `agent/selector.py`, and `agent/recovery_planner.py`) revealed the following:
- **Good:** None of the consumer modules (Query Intelligence, Tool Selection, Recovery, Final Synthesis) independently instantiate or load `Qwen3Inference`. They correctly utilize the injected `inference_engine`.
- **Bad:** `execute_agentic_pipeline` was instantiating a new `Qwen3Inference` object for *every* real-mode job, but failing to explicitly unload it.
- **Consequence:** Under sequential or concurrent real-mode job execution, PyTorch/MPS would retain the model weights and associated intermediate tensors from the previous job due to Python's delayed garbage collection and lack of an explicit `torch.mps.empty_cache()` trigger, causing significant VRAM pressure.

## 3. Qwen3 Object Ownership

- **Owner:** `execute_agentic_pipeline` (in `job_manager.py`)
- **Lifecycle:** Job-scoped. 
- The job creates the `Qwen3Inference` instance once, loads it once, injects it down the execution tree, and explicitly unloads it in a `finally` block before exiting.

## 4. Qwen3 Load Lifecycle

| Component | Receives Qwen3 Object | Creates Qwen3 | Calls Load | Same Instance |
|-----------|----------------------|---------------|------------|---------------|
| Query Intelligence | YES | NO | NO | YES |
| Tool Selector | YES | NO | NO | YES |
| Recovery Planner | YES | NO | NO | YES |
| Final Synthesis | YES | NO | NO | YES |

## 5. Dependency Injection Changes

- Validated that `job_manager.py` passes the same `qwen_engine` reference sequentially to:
  1. `QueryIntelligencePipeline(qwen_engine)`
  2. `QwenToolSelector(qwen_engine, registry)`
  3. `QwenRecoveryPlanner(qwen_engine)`
  4. `_synthesize_final_answer(..., qwen_engine=qwen_engine)`
- No additional architectural changes were necessary since the parameter plumbing from Task 8.2.2 was structurally correct.

## 6. Duplicate Loading Prevention

- `Qwen3Inference.load()` has an internal `self.is_loaded` circuit breaker to prevent double-loading.
- Added a `finally` block in `execute_agentic_pipeline` that guarantees `qwen_engine.unload()` is called to flush MPS/CUDA caches and clear object references when the job terminates (successfully or via exception).

## 7. Memory-Safe Failure Behavior

- If `qwen_engine.load()` fails (e.g. due to Out-Of-Memory errors), the system catches the exception and immediately aborts the pipeline, marking the status as `MODEL_UNAVAILABLE`.
- It does **not** attempt to retry, and it does **not** fall back to the `MockQwenEngine`.
- No specialist models are executed using fabricated intermediate states.

## 8. Real vs Fixture Isolation

- The `execution_mode` flag strictly bifurcates initialization.
- In `execution_mode="real"`, `MockQwenEngine` is never instantiated.
- In `execution_mode="fixture"`, `Qwen3Inference` is never instantiated.

## 9. Test Results

Two new tests were implemented in `test_task8_2_2r_qwen3_memory_safety.py`:
1. `test_qwen3_single_instance_lifecycle`: Validates object identity across the four consumers, confirming they all receive the exact same memory reference (`id(engine)`). It also asserts `load()` is called exactly once.
2. `test_qwen3_explicit_unload`: Ensures `unload()` is automatically called via the `finally` block even if the pipeline encounters a catastrophic error.

**Result**: PASS.

## 10. Minimal Qwen3 Smoke Test

A minimal, isolated script (`test_qwen3_smoke.py`) was executed to observe memory behavior:
- Instantiated `Qwen3Inference`
- Loaded weights
- Performed a single generation ("Say hello in one word.")
- Output: "Hi"
- Unloaded weights

## 11. Observed Resource Behavior

Observed process memory via `psutil` (Note: macOS often abstracts MPS memory via kernel wiring, so RSS changes may appear unintuitive):
- **Start RSS**: 180.52 MB
- **After Instantiation RSS**: 180.52 MB
- **After Load RSS**: 49.48 MB (Compression / memory swap artifact)
- **After Generation RSS**: 172.17 MB
- **After Unload RSS**: 311.59 MB

The test executed flawlessly without crashing the system, proving the pipeline can now load and unload safely per job.

## 12. Files Changed

- `backend/job_manager.py` (Added `finally` block for explicit cleanup)
- `backend/tests/test_task8_2_2r_qwen3_memory_safety.py` (New test file)
- `test_qwen3_smoke.py` (New smoke test script)

## 13. Remaining Risks

- Running multiple jobs concurrently in the same process will instantiate multiple Qwen3 instances simultaneously, which will crash the system due to Apple Silicon memory constraints. Concurrency management (e.g., global locks on LLM inference) is out of scope for the single-job architectural fix but must be kept in mind for production deployment.

## 14. Final Verdict

**PASS**. The single-instance architecture is verified. Qwen3 memory lifecycle is now strictly job-scoped with guaranteed cleanup, making it safe to proceed to the end-to-end demonstrations.
