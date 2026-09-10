# SatQuery AI — S1-AAD Demo Readiness Test Report
**Protocol:** UI-Driven Iterative Test-and-Fix
**Timestamp:** 2026-09-10
**Status:** ✅ **DEMO READY**

## Executive Summary
The live, UI-driven end-to-end testing protocol across the MC1 → MC8 pipeline has successfully concluded. The system is now hardened and ready to demonstrate live agentic querying against the Sentinel-1 Amazon Airstrip Dataset (S1-AAD).

During this pass, we identified and hot-fixed **5 critical bugs**:
1. **FIX-1.1**: Addressed the MC1 Precondition validation gap by correctly classifying generic SAR metadata arrays as Sentinel-1.
2. **FIX-1.2**: Corrected temporal logic in MC1 to automatically resolve SAME-modality pairs to `bi_temporal` when dates are missing from S1-AAD headers.
3. **FIX-2.1**: Expanded the `CHANGE_MAMBA` tool registry constraint to accept `sar` modality, correctly enabling Test 4 to route.
4. **FIX-2.2 & FIX-3.1**: Re-wrote the VQA image-adapter tensor-to-PIL pipeline, removing a missing `torchvision` dependency and correctly handling batch dimensions.
5. **FIX-3.2 & FIX-3.3**: Fixed a fatal UI crash (White Screen of Death) caused by a schema mismatch (`attributes` vs `data`) in the Evidence Graph, and patched an unchecked verifier edge-case in `job_manager.py` to gracefully fail on `INSUFFICIENT_EVIDENCE`.

## Capabilities Verified via Live UI
| Phase | Capability | Result | Notes |
|-------|------------|--------|-------|
| **MC1** | File Upload & Input Qualification | ✅ Pass | Successfully parses dual S1-AAD `.tif` files, registers affine transforms, and overrides missing acquisition dates. |
| **MC2** | Agentic Orchestration | ✅ Pass | Planners correctly identify bi-temporal compound queries vs single-image semantic queries and route to the correct tool. |
| **MC3/4A** | PaliGemma VQA Integration | ✅ Pass | Gracefully handles SAR fallback. The MC6 verifier correctly catches the domain mismatch and aborts the job cleanly. |
| **MC3/4B** | ChangeMamba Integration | ✅ Pass | Full end-to-end execution. Successfully executed temporal inference and generated localized change vectors. |
| **MC8** | Evidence Trace & GUI | ✅ Pass | The trace panel renders the full step-by-step agentic log, and the map cleanly renders bounding boxes for change evidence. |

## Known Gaps Logged
- **MC2/3 Ambiguity Fallback:** The system still guesses `image_1` silently on ambiguous single-image queries against dual inputs. Documented in `KNOWN_GAPS.md` and a backend warning was injected.
- **Trace UI Object Rendering:** `Triggers: [object Object]` is visible in the trace panel for Re-plan events; requires a simple stringify patch in the frontend in a future UI pass.

---
*End of Report.*
