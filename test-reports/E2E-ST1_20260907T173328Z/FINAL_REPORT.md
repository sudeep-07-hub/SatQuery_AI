# SatQuery AI — End-to-End System Test Protocol (E2E-ST1)
**Date:** 2026-09-07
**Dataset:** S1-AAD Sentinel-1 Amazon Airstrip Dataset

## 1. Executive Summary

SatQuery AI currently operates with severe validation vulnerabilities that prevent a reliable end-to-end execution on real data. While the system can successfully inventory and extract metadata from single chips and bi-temporal pairs, it **fails to enforce critical spatial constraints**. Specifically, when fed two entirely disconnected images with non-overlapping footprints, the Geo-Input Qualification layer (MC1) silently accepts the pairing rather than rejecting it, flagging the job as executable. Because of this critical failure at the very first gate, downstream components (MC2, MC4) cannot be reliably tested in an automated fashion, as they would be receiving corrupt/invalid data topologies. **The system cannot yet be trusted to safely route and execute tasks until the MC1 validation gate is hardened.** Additionally, the system currently lacks MC3, MC4C, MC5, MC6, and MC7, meaning it does not yet perform cross-specialist evidence reconciliation, closed-loop re-planning, or final answer synthesis. 

## 2. Phase-by-Phase Status

| Phase | Description | Status | Tests Passed | Tests Failed |
| :--- | :--- | :--- | :--- | :--- |
| **0** | Dataset & Codebase Inventory | **PASS** | 7/7 | 0 |
| **1** | MC1 — Geo-Input Qualification, Real Data | **FAIL** | 5/6 | 1 |
| **2** | MC2 — Orchestration Controller | **BLOCKED** | - | - |
| **3** | MC4A — PaliGemma VQA, Real Chips | **BLOCKED** | - | - |
| **4** | MC4B — ChangeMamba Bi-Temporal, Real Pairs | **BLOCKED** | - | - |
| **5** | MC8 — Full UI / Export Round Trip | **BLOCKED** | - | - |

*(Note: Per the strict test protocol's failure policy, a critical rejection failure in Phase 1 immediately blocks execution of subsequent phases to prevent compounding errors downstream).*

## 3. Critical Findings

> [!CAUTION]
> **Silent Validator Pass-Through on Non-Overlapping Inputs**
> - **Component:** MC1 (`backend/mc1/pipeline.py`)
> - **Issue:** When providing two valid but non-overlapping TIFF files (e.g., `_ID_100.tif` and `_ID_104.tif`), MC1 correctly computes `spatial_overlap = 0.0`. However, it does **not** append a fatal warning to the profile. 
> - **Consequence:** Because there are no fatal warnings, `task_executable` remains `True`. The Orchestration Controller (MC2) will erroneously receive this as a valid pair and dispatch it to the ChangeMamba engine, which expects perfectly co-registered, overlapping chips. This will lead to catastrophic hallucination or model crashes.

## 4. Minor Findings

- None logged (testing aborted at Phase 1).

## 5. Known Gaps

> [!NOTE]
> The following are accepted current-scope limitations and are **not** test failures:
> - **MC2 `target_image` Schema Gap:** The multi-image disambiguation for single-image queries is an open architectural gap.
> - **MC3, MC4C, MC5, MC6, MC7 Absent:** The system does not yet possess the registry/planner decoupling, optical-SAR fusion, evidence normalization, verification loops, or answer generation modules.

## 6. Prioritized Recommendations

1. **[IMMEDIATE]** Patch `backend/mc1/pipeline.py` to assert that `spatial_overlap > 0` (or a specific threshold like `0.95`). If the overlap is insufficient, append a fatal warning so that `task_executable` becomes `False`.
2. **[IMMEDIATE]** Once MC1 is patched, re-run this exact E2E-ST1 test protocol from scratch to unblock Phases 2 through 5.
3. **[DEFERRED]** Implement the `target_image` schema field in MC2 to support VQA disambiguation over multi-image inputs.
4. **[DEFERRED]** Begin development on MC2 Query Understanding and Task Decomposition.

## 7. Appendix

- [Phase 0/1 Log](file:///Users/sukesh/Desktop/satquery/test-reports/E2E-ST1_20260907T173328Z/phase_log.md)
- [Summary JSON](file:///Users/sukesh/Desktop/satquery/test-reports/E2E-ST1_20260907T173328Z/summary.json)
- [Test Script](file:///Users/sukesh/Desktop/satquery/test-reports/E2E-ST1_20260907T173328Z/test_mc1.py)
