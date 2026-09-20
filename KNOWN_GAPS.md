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

## 8. Horizontal Overflow on Narrow Screens (Frontend) — RESOLVED
- **Symptom**: `/assistant` scrolls horizontally on a phone. `document.scrollWidth` measures 401 px at both a 320 px and a 375 px viewport.
- **Cause**: the chat header's `ModelSelector` is a native `<select>` sized by its widest option ("More controllers — coming soon (not available)"): 329 px inside a 400 px label that does not shrink. The composer row itself fits at 320 px.
- **Not caused by the composer work** — found while verifying composer alignment at 320 px (FIX_LOG.md, 2026-09-20) and left alone, as the header is outside that change's scope.
- **Required fix**: let the control shrink (`min-width: 0` on the label, a `max-width`/truncated trigger), or replace the native `<select>` — it is a placeholder with exactly one real option today.
- **RESOLVED 2026-09-20** (Phase 2): fixed with `min-width: 0; max-width: 100%` on the selector and its `<select>`, and the chat-header capability pill now wraps on narrow screens. Verified at 320 px in all five interface languages, light and dark: no horizontal scroll, nothing clipped. The original entry above is kept as the record of what was found. (The 401 px figure was measured with the backend down; with a backend reachable the capability pill was a second, larger cause at 445 px English / 531 px Tamil.)

## 9. Interface Translation Quality — OPEN
- The Hindi, Kannada, Telugu and Tamil catalogs (`frontend/src/i18n/`) were produced without native-speaker review. Key parity and rendering are verified; **wording is not**.
- The risky set is remote-sensing vocabulary, which has no settled everyday equivalent in these languages. The specific strings needing review are listed in FIX_LOG.md's Phase 2 gate; the recurring terms are *change detection*, *evidence*, *footprint overlap*, *georeferenced*, *bi-temporal pair*, *uncalibrated confidence*, *abstain* and *verification*.
- **Required fix**: a speaker of each language reviews the catalog before this is shown as finished multilingual support.

## 10. Language Scope Is the Interface Only — OPEN
- Phase 2 translates interface chrome. Three things stay English by design, and the UI does not hide it:
  - The query POSTed to `/api/query`, including the text on the sample cards — the planner and the Tool Registry rules match English, so translating the query would silently change what the pipeline reasons over. That is Phase 3 and is separately gated.
  - Backend-generated answer prose, caveats and failure reasons.
  - Engine display names on evidence chips (e.g. "Classical change detection"), which are kept verbatim so a chip matches the `source_model` in the execution trace.
- **Required fix**: Phase 3 of the multilingual work order.

## 11. Chat History Does Not Retranslate — OPEN
- Assistant turns are persisted to `localStorage` as text. Client-generated messages (backend unreachable, job missing, analysis failed) are written in the language that was active at the time, so a language switch leaves older turns in the previous language. Live interface chrome retranslates correctly.
- **Required fix**: persist a translation key plus parameters instead of rendered text, and translate at render time — a `chatStorage` schema change with a migration for stored sessions.

## 12. Seven Unreferenced Components — OPEN
- `ResultPanel`, `TracePanel`, `QueryBox`, `ProfilePanel`, `TestHarness`, `AnalyzeButton` and `JobStatus` are imported by nothing (only their own definitions reference their names). They are leftovers from the pre-chat single-page UI.
- They were deliberately **not** translated in Phase 2 — roughly 35 strings × 5 languages of dead weight — and they still contain hardcoded English.
- They also reference two CSS custom properties that were never defined, `--border-light` and `--shadow-sm`, so those declarations have always been dropped as invalid. `--shadow-sm` is now defined (the language popover is its first live user); `--border-light` is still undefined.
- **Required fix**: delete them, or bring them back into the app and finish translating them. Deleting files was outside the multilingual work order's scope.

## 13. Execution Trace Stays English — OPEN (by design)
- The execution trace is an audit artefact and is not translated: stage names, tool ids, planner fields, verification keys and the backend's own stage messages stay English in every interface language. The UI says so with a translated note at the end of the trace.
- The one exception is deliberate: when a query was translated, the trace's first section shows **both** the original question and the English string the pipeline actually reasoned over (`input_translation`), so a reviewer can check the translation rather than trust it.
- **Required fix**: none for auditability. If trace prose is ever wanted in-language, translate it for display only and keep the English as the stored value.

## 14. Exports Are English Only — OPEN
- PDF, GeoJSON and JSON exports are written from the English originals (`final_answer_en`, `caveats_en`) even when the answer was shown in another language, and the result carries `exports_language: "en"`.
- **Cause**: ReportLab's built-in Type 1 faces have no Devanagari, Kannada, Telugu or Tamil glyphs, so an Indic PDF would be a page of empty boxes. Embedding a TTF per script was not done — it needs four font files shipped with the backend and a font-registration path in `mc8_export/exporter.py`.
- **Required fix**: register and embed a Unicode TTF per script (e.g. the Noto family already used by the frontend) and verify with a real generated PDF in each language; or keep English exports and state it in the UI at the download.

## 15. Non-English Queries Need the Language Model — OPEN (by design)
- When Qwen3 is unreachable and the planner falls back to Tool Registry rules, a non-English query terminates as `TRANSLATION_UNAVAILABLE` with no claims and zero confidence, and the UI shows the backend's reason. English queries on the same deployment are unaffected.
- This is deliberate: the registry matches English keywords, so running it on another language would produce a confident answer to a question the pipeline never understood.
- **Consequence for the free-tier deployment**: the Render blueprint sets `SATQUERY_QWEN_BACKEND=none`, so that deployment serves English queries only.
- **Required fix**: none, unless a translation path that does not need the LLM is wanted — which would mean adding a translation dependency, which this work order forbids.

## 16. Query and Answer Translation Quality — OPEN
- Query translation into English, and answer/caveat translation out of English, are done by Qwen3-4B (Q4_K_M) with no human in the loop and no review. The prompt instructs it to preserve hedges, limitations, numbers and identifiers, but nothing enforces that.
- The risk is specific and worth stating: an answer's caveats are what keep it honest. A translation that softens "the evidence cannot confirm whether a new airstrip has been cleared" would misrepresent the system. The English original is always kept as `final_answer_en` / `caveats_en`, and the trace shows the translated query, so the drift is at least auditable.
- **Measured across the four non-English languages on one real job** (2 caveats each, 8 passages): **5 translated, 3 returned in English**. Hindi 2/2, Telugu 2/2, Tamil 1/2, **Kannada 0/2**. Answer prose and query translation succeeded in all four. So a Kannada user currently sees a Kannada answer with English caveats.
- **Measured behaviour**, not speculation. On real caveats from a real run: a short one translated correctly and kept its identifiers (`image_1 = पहले, image_2 = बाद`); a long technical one could not be translated and came back in English. A batched "translate this numbered list" prompt was tried and abandoned — at temperature 0.0 the model looped one phrase until the JSON was unterminated, and with a repetition penalty it returned the *same invented sentence* for every passage. Translation is therefore one passage per call, and anything that fails comes back in English rather than as something plausible and wrong.
- **Consequences to expect**: a mix of languages within one answer's caveat list, and extra latency (a non-English job makes 1 + 1 + N model calls, where N is the number of caveats).
- **Required fix**: native-speaker review of real translated answers, and a check that hedging survives; or a fidelity check comparing the translated answer back against the English; or a larger/instruction-tuned translation model, which would mean a new dependency.

## 17. CROMA Test Modules Cannot Be Collected In This Environment — OPEN
- `tests/test_mc4c_engine.py`, `test_mc4c_fallback.py`, `test_mc4c_semantic_head.py` and `test_task7_2_croma.py` fail at import with `ModuleNotFoundError: No module named 'configilm'`. `mc4c/semantic_head.py` imports it, but it is in neither `requirements.txt` nor `requirements-render.txt`.
- Pre-existing and unrelated to the multilingual work; it means a plain `pytest` run aborts on collection unless `--continue-on-collection-errors` is passed, and the full suite count cannot be reproduced on a clean checkout.
- **Required fix**: add `configilm` to `requirements.txt`, or guard the import in `mc4c/semantic_head.py` the way the other optional engines are guarded.

## 18. Evaluation Runner Writes To A Repo-Root-Relative Path — OPEN
- `backend/evaluation/runner.py:11` defaults `output_dir="backend/data/reports"`. That resolves correctly when the process starts at the repository root, but the test suite runs from `backend/`, so a full `pytest` run creates a stray **`backend/backend/data/reports/`** holding `eval_rsvqa_fixture.json`, `eval_vrsbench_fixture.json` and `eval_cdvqa_fixture.json`.
- The stray directory is not covered by `.gitignore`, so it would be committed by a `git add -A` after running the tests.
- Pre-existing and unrelated to the multilingual work; found while running the suite for the Phase 3 gate, and deleted rather than committed.
- **Required fix**: resolve the path relative to the module (`Path(__file__).resolve().parents[1] / "data" / "reports"`) instead of the current working directory.
