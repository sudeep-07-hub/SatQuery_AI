"""
job_manager.py — Async execution manager for the Full Agentic Architecture.

Integrates MC1–MC8 + Qwen3 4B into a single coherent end-to-end workflow:

    USER QUERY + OBSERVATIONS
        → MC1 Input Qualification (+ loading the uploaded rasters)
        → Query Intelligence (Qwen3, or deterministic registry rules)
        → MC3 Tool Selection (availability-aware Tool Registry) + Binding + Planning
        → AgentController + RecoveryManager (bounded execution)
        → MC5 Evidence Normalization + Evidence Graph
        → MC6 Verification Gate (per-evidence)
        → Final Answer Synthesis (Qwen3, or evidence-only template)
        → MC8 Export

Planner / LLM configuration (environment):
    SATQUERY_QWEN_BACKEND      ollama (default) | transformers | none
    SATQUERY_PLANNER_FALLBACK  registry (default): use Tool Registry rules if Qwen3 cannot run
                               strict: stop with MODEL_UNAVAILABLE instead
"""
import asyncio
import os
import shutil
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional, List
import torch
import traceback
import logging

from mc1.pipeline import run_mc1_pipeline
from mc1.schemas import (
    RequestObservationProfile, ObservationProfile,
    SensorProfile, SpatialProfile, QualityProfile, CompatibilityProfile
)
from mc5_evidence.evidence_graph import EvidenceGraph
from mc6_verification.verifier import Verifier

from qwen.pipeline import QueryIntelligencePipeline, QueryIntelligenceResult
from qwen.schemas import TaskSpec, ObservationRequirement, SubtaskSpec
from qwen import engine_factory
from agent.default_tools import setup_default_registry
from agent.selector import QwenToolSelector
from agent.binding import ObservationBinder
from agent.planner import WorkflowPlanner
from agent.controller import AgentController
from agent.recovery import RecoveryManager
from agent.recovery_planner import QwenRecoveryPlanner
from agent.adapters import build_adapters
from mc4b_temporal.classical import image_bounds_wgs84
from agent.rule_based import (
    RegistryQueryInterpreter, FallbackToolSelector, FallbackRecoveryPlanner, template_answer
)

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = [
    "DONE", "FAILED", "ABSTAIN", "INSUFFICIENT_EVIDENCE", "PRECONDITION_FAILED",
    "MODEL_UNAVAILABLE", "INSUFFICIENT_OBSERVATIONS",
]


class JobRegistry:
    """
    In-memory job store. Finished jobs keep their loaded rasters, so on small hosts the number of
    retained jobs is capped (SATQUERY_MAX_JOBS, 0 = unlimited): the oldest finished jobs are evicted
    together with their uploaded files and exports.
    """
    def __init__(self, max_jobs: Optional[int] = None):
        self._jobs: Dict[str, Dict] = {}
        self.max_jobs = max_jobs if max_jobs is not None else int(os.getenv("SATQUERY_MAX_JOBS", "0"))

    def _evict_finished(self):
        if self.max_jobs <= 0:
            return
        finished = [jid for jid, job in self._jobs.items() if job["status"] in TERMINAL_STATUSES]
        while len(self._jobs) >= self.max_jobs and finished:
            jid = finished.pop(0)  # dicts keep insertion order: oldest first
            job = self._jobs.pop(jid)
            paths = list((job.get("exports") or {}).values())
            if job.get("change_overlay_png"):
                paths.append(job["change_overlay_png"])
            for path in paths:
                try:
                    os.remove(path)
                except OSError:
                    pass
            shutil.rmtree(os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads", jid), ignore_errors=True)

    def create_job(self) -> str:
        self._evict_finished()
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = {
            "job_id": job_id,
            "status": "QUEUED",
            "progress_trace": [],
            "result": None,
            "evidence_graph": None,
            "mc1_profile": None,
            "exports": {},
            "rasters": {},
        }
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict]:
        return self._jobs.get(job_id)

    def update_status(self, job_id: str, status: str, trace_entry: Optional[Dict] = None):
        if job_id in self._jobs:
            self._jobs[job_id]["status"] = status
            if trace_entry:
                trace_entry["timestamp"] = datetime.now(timezone.utc).isoformat()
                trace_entry["stage"] = status
                self._jobs[job_id]["progress_trace"].append(trace_entry)

job_registry = JobRegistry()

# Static registry with every capability enabled (fixture mode and legacy callers).
registry = setup_default_registry()
for capability in registry.list_capabilities():
    capability.enabled = True

adapters = build_adapters("real")


# Capabilities switched off for a deployment regardless of local model availability (e.g. on a
# 512 MB host where loading PaliGemma would exhaust memory): comma-separated tool ids.
DISABLED_TOOLS = {t.strip() for t in os.getenv("SATQUERY_DISABLED_TOOLS", "").split(",") if t.strip()}


def build_job_registry(execution_mode: str, job_adapters: Dict) -> tuple:
    """
    Per-job Tool Registry. In real mode each capability is enabled only if its engine can run here,
    so tool selection never routes to an engine that is known to be unavailable.
    Returns (registry, availability report).
    """
    job_reg = setup_default_registry()
    report = []
    for capability in job_reg.list_capabilities():
        adapter = job_adapters.get(capability.tool_id)
        if execution_mode == "fixture":
            available, reason = True, "fixture mode"
        elif adapter is None:
            available, reason = False, "no execution adapter"
        elif capability.tool_id in DISABLED_TOOLS:
            available, reason = False, "DISABLED: turned off for this deployment (SATQUERY_DISABLED_TOOLS)"
        else:
            available, reason = adapter.availability()
        capability.enabled = available
        if not available:
            capability.status = "unavailable"
        report.append({"tool_id": capability.tool_id, "name": capability.name, "available": available, "reason": reason})
    return job_reg, report


class MockQwenEngine:
    """Deterministic mock for environments where Qwen3 weights are not loaded (fixture mode only)."""
    def load(self):
        pass

    def generate(self, prompt, **kwargs):
        if "You are a query decomposition engine" in prompt:
            # Check if this is the multitool query
            if "determine if there are changes" in prompt and "Describe the scene" in prompt:
                # Mock a compound result
                return {"status": "ok", "response": '''{
                  "is_compound": true,
                  "ambiguous": false,
                  "subtasks": [
                    {
                      "subtask_id": "subtask_1",
                      "description": "Describe the scene in optical",
                      "primary_task": "single_image_vqa",
                      "target_entities": [],
                      "required_modalities": ["optical"],
                      "temporal_requirement": "none",
                      "spatial_output_required": false,
                      "textual_output_required": true,
                      "depends_on": []
                    },
                    {
                      "subtask_id": "subtask_2",
                      "description": "Determine if there are changes between the two dates",
                      "primary_task": "change_detection",
                      "target_entities": [],
                      "required_modalities": ["optical"],
                      "temporal_requirement": "before_after",
                      "spatial_output_required": false,
                      "textual_output_required": true,
                      "depends_on": ["subtask_1"]
                    }
                  ]
                }'''}
            return {"status": "ok", "response": '{"is_compound": false}'}

        if "You are a remote-sensing analysis assistant" in prompt:
            # Extract evidence IDs from prompt to mock a valid response
            import re
            evidence_ids = re.findall(r"\[([a-zA-Z0-9_-]+)\]", prompt)
            ids_str = ", ".join(f'"{eid}"' for eid in evidence_ids)
            return {"status": "ok", "response": f'{{"answer": "Mock synthesis", "evidence_ids": [{ids_str}]}}'}

        if "Primary Intent: cross_modal_fusion" in prompt:
            return {"status": "ok", "response": '{"tool_id": "optical_sar_fusion", "arguments": {"query": "Perform fusion"}}'}
        elif "Primary Intent: change_detection" in prompt or "Requested Task Profile:\\n- Primary Intent: change_detection" in prompt:
            return {"status": "ok", "response": '{"tool_id": "temporal_change_analysis", "arguments": {"query": "Detect changes"}}'}

        return {"status": "ok", "response": '{"tool_id": "single_image_vqa", "arguments": {"query": "Fallback"}}'}

mock_engine = MockQwenEngine()
binder = ObservationBinder()
planner = WorkflowPlanner()
controller = AgentController(registry, adapters)

# MC6 Verifier
verifier = Verifier()


def _build_observation_profiles(mc1_profile_legacy: Dict) -> List[ObservationProfile]:
    """Convert legacy MC1 dict into formal ObservationProfile list."""
    obs_list = []
    for key in ["image_1", "image_2"]:
        img = mc1_profile_legacy.get(key)
        if not img:
            continue
        obs_list.append(ObservationProfile(
            observation_id=key,
            file_source=img.get("filename", f"{key}.tif"),
            file_format="tiff",
            spatial=SpatialProfile(
                crs=img.get("crs", "EPSG:4326"),
                gsd_m=img.get("gsd_m", 1.0)
            ),
            sensor=SensorProfile(
                modality=img.get("modality", "optical"),
                sensor=img.get("sensor", "unknown")
            ),
            quality=QualityProfile(score=img.get("quality_score", 1.0)),
            temporal={"timestamp": img.get("acquisition_date")} if img.get("acquisition_date") else None
        ))
    return obs_list


def _partition_evidence(evidence: List[Dict], verification_result: Dict):
    """Split evidence into verified / rejected using MC6 triggers that name a specific evidence_id."""
    rejected_ids = {
        t.get("evidence_id") for t in verification_result.get("triggers_fired", [])
        if isinstance(t, dict) and t.get("evidence_id")
    }
    verified = [ev for ev in evidence if ev.get("evidence_id") not in rejected_ids]
    rejected = [ev for ev in evidence if ev.get("evidence_id") in rejected_ids]
    global_triggers = [
        t for t in verification_result.get("triggers_fired", [])
        if not (isinstance(t, dict) and t.get("evidence_id"))
    ]
    return verified, rejected, global_triggers


def _synthesize_final_answer(
    query: str,
    evidence: List[Dict],
    verification_result: Dict,
    agent_state_status: str,
    model_unavailable_tools: List[str],
    execution_mode: str = "real",
    qwen_engine=None,
    limitations: Optional[List[str]] = None,
) -> Dict:
    """
    Synthesize the final answer from verified evidence using Qwen3, or an evidence-only
    template when no LLM is available. Never loads a model on its own.
    """
    import json

    # Determine execution-level status
    if verification_result["status"] == "VERIFIED":
        execution_status = "SUCCESS"
    elif verification_result["status"] == "INSUFFICIENT_EVIDENCE":
        execution_status = "INSUFFICIENT_EVIDENCE"
    elif verification_result["status"] == "RE_PLAN_REQUIRED":
        execution_status = "VERIFICATION_FAILURE"
    else:
        execution_status = agent_state_status.upper()

    claims = [ev.get("claim", "") for ev in evidence]
    evidence_ids = [ev.get("evidence_id", "") for ev in evidence]
    confidence = max([ev.get("confidence", 0.0) for ev in evidence]) if evidence else 0.0

    # Handle MODEL_UNAVAILABLE distinctly
    if model_unavailable_tools:
        unavailable_names = ", ".join(model_unavailable_tools)
        answer_text = (
            f"The query could not be fully answered because the following specialist "
            f"model(s) are unavailable in the current environment: {unavailable_names}. "
            f"No fabricated results have been produced."
        )
        return {
            "final_answer": answer_text,
            "claims": claims,
            "evidence_references": [],
            "confidence": confidence,
            "execution_status": "MODEL_UNAVAILABLE",
            "verification_status": verification_result["status"],
            "model_unavailable": model_unavailable_tools,
            "answer_generator": "template",
        }

    if not evidence:
        rejected = verification_result.get("rejected_claims") or []
        if rejected:
            message = (
                f"The specialist produced {len(rejected)} answer(s), but all were rejected by verification "
                f"(confidence below the {Verifier().confidence_threshold} threshold), so no answer is given. "
                + " ".join(f"Rejected: \"{r.get('claim', '')}\" (confidence {r.get('confidence', 0):.2f})." for r in rejected)
            )
        else:
            message = "No evidence was generated by any specialist model. The query cannot be answered."
        return {
            "final_answer": message,
            "claims": [],
            "evidence_references": [],
            "confidence": 0.0,
            "execution_status": "INSUFFICIENT_EVIDENCE",
            "verification_status": verification_result["status"],
            "model_unavailable": model_unavailable_tools,
            "answer_generator": "template",
        }

    rejected_claims = verification_result.get("rejected_claims", [])
    answer_generator = "template"

    if qwen_engine is not None and hasattr(qwen_engine, 'generate'):
        prompt = (
            f"You are a remote-sensing analysis assistant. "
            f"Answer the user's query using ONLY the verified evidence below. "
            f"Do not invent details not present in the evidence. "
            f"Do not invent spatial locations or quantitative values. "
            f"If the evidence cannot confirm what the user asked about (for example the type of object), say so explicitly. "
            f"Acknowledge uncertainty where appropriate.\n\n"
            f"Query: {query}\n\n"
            f"Verified Evidence:\n"
        )
        for ev in evidence:
            prompt += f"- [{ev.get('evidence_id')}] {ev.get('claim', 'N/A')} (confidence: {ev.get('confidence', 0.0):.2f}, source: {ev.get('source_model', 'unknown')})\n"

        if rejected_claims:
            prompt += f"\nRejected Claims (DO NOT TREAT AS FACT):\n"
            for r_claim in rejected_claims:
                prompt += f"- [{r_claim.get('evidence_id', 'unknown')}] {r_claim.get('claim', 'N/A')}\n"

        if limitations:
            prompt += "\nKnown limitations (must be reflected in the answer):\n"
            for lim in limitations:
                prompt += f"- {lim}\n"

        prompt += (
            f"\nRespond in strict JSON format matching this schema:\n"
            f"{{\n"
            f'  "answer": "your grounded answer here",\n'
            f'  "evidence_ids": ["ev_1", "ev_2", ...],\n'
            f'  "uncertainty": "any uncertainty or missing information"\n'
            f"}}\n"
        )

        try:
            result = qwen_engine.generate(prompt)
            if result.get("status") == "ok":
                raw_response = result["response"]
                try:
                    json_str = raw_response
                    if "```json" in json_str:
                        json_str = json_str.split("```json")[1].split("```")[0]
                    elif "```" in json_str:
                        json_str = json_str.split("```")[1].split("```")[0]

                    parsed = json.loads(json_str.strip())
                    answer_text = parsed.get("answer", raw_response)
                    gen_ids = parsed.get("evidence_ids", [])
                    valid_ids = [eid for eid in gen_ids if eid in evidence_ids]
                    if parsed.get("uncertainty"):
                        answer_text = str(answer_text).strip()
                        if answer_text and answer_text[-1] not in ".!?":
                            answer_text += "."
                        answer_text = f"{answer_text} Uncertainty: {parsed['uncertainty']}"
                except json.JSONDecodeError:
                    answer_text = raw_response
                    valid_ids = evidence_ids
                if execution_mode == "fixture":
                    answer_text = f"[TEST FIXTURE] {answer_text}"
                answer_generator = "qwen3"
            else:
                answer_text = template_answer(query, evidence, rejected_claims)
                valid_ids = evidence_ids
        except Exception:
            answer_text = template_answer(query, evidence, rejected_claims)
            valid_ids = evidence_ids
    else:
        answer_text = template_answer(query, evidence, rejected_claims)
        valid_ids = evidence_ids

    return {
        "final_answer": answer_text,
        "claims": claims,
        "evidence_references": valid_ids,
        "confidence": confidence,
        "execution_status": execution_status,
        "verification_status": verification_result["status"],
        "model_unavailable": model_unavailable_tools,
        "answer_generator": answer_generator,
    }


def _finish_early(job_id: str, status: str, message: str, result: Dict, trace_extra: Optional[Dict] = None):
    job = job_registry.get_job(job_id)
    job["result"] = {
        "claims": [],
        "evidence_references": [],
        "confidence": 0.0,
        **result,
    }
    job_registry.update_status(job_id, status, {"message": message, **(trace_extra or {})})
    from mc8_export.exporter import generate_exports
    job["exports"] = generate_exports(job)


def _fixture_smoke_profile(mc1_profile_legacy: Dict, query: str, files_data: List[tuple]) -> None:
    """Fixture mode only: overwrite MC1 output with a deterministic synthetic profile."""
    mc1_profile_legacy["task_executable"] = True
    mc1_profile_legacy["spatial_overlap"] = 0.95
    mc1_profile_legacy["coregistration_score"] = 0.92

    is_fusion = "fusion" in query.lower() or "cross_modal" in query.lower()

    mc1_profile_legacy["image_1"] = {
        "filename": "img1.tif", "modality": "optical",
        "crs": "EPSG:32643", "gsd_m": 1.0,
        "acquisition_date": "2020-01-01T00:00:00Z"
    }
    if len(files_data) > 1:
        mc1_profile_legacy["image_2"] = {
            "filename": "img2.tif", "modality": "sar" if is_fusion else "optical",
            "crs": "EPSG:32643", "gsd_m": 1.0,
            "acquisition_date": "2020-01-01T00:00:00Z" if is_fusion else "2024-01-01T00:00:00Z"
        }
    mc1_profile_legacy["affine_transform"] = [1.0, 0, 300000.0, 0, -1.0, 4000000.0]
    mc1_profile_legacy["image_count"] = len(files_data)


def _fixture_smoke_intent(query: str) -> QueryIntelligenceResult:
    if "compound" in query.lower():
        return QueryIntelligenceResult(
            original_query=query,
            primary_task_spec=TaskSpec(
                query=query, primary_task="change_detection", target_entities=[], required_modalities=["optical"],
                spatial_output_required=True, textual_output_required=True, temporal_requirement="before_after"
            ),
            is_compound=True,
            subtasks=[
                SubtaskSpec(
                    subtask_id="subtask_1", description="Describe the scene in optical", primary_task="single_image_vqa",
                    target_entities=[], required_modalities=["optical"], temporal_requirement="none",
                    spatial_output_required=False, textual_output_required=True
                ),
                SubtaskSpec(
                    subtask_id="subtask_2", description="Determine if there are changes between the two dates",
                    primary_task="change_detection", target_entities=[], required_modalities=["optical"],
                    temporal_requirement="before_after", spatial_output_required=True, textual_output_required=True
                )
            ],
            observation_requirements=[
                ObservationRequirement(
                    requirement_id="req_1", source_subtasks=[], minimum_observations=2, maximum_observations=2,
                    required_modalities=["optical"], temporal_relationship="before_after"
                )
            ],
            ambiguous=False
        )
    if "fusion" in query.lower() or "cross_modal" in query.lower():
        primary_task, req_mods, temp_req = "cross_modal_fusion", ["optical", "sar"], "none"
    elif "change" in query.lower():
        primary_task, req_mods, temp_req = "change_detection", ["optical"], "before_after"
    else:
        primary_task, req_mods, temp_req = "single_image_vqa", ["optical"], "none"

    spatial = primary_task in ["change_detection", "cross_modal_fusion"]
    return QueryIntelligenceResult(
        original_query=query,
        primary_task_spec=TaskSpec(
            query=query, primary_task=primary_task, target_entities=[], required_modalities=req_mods,
            spatial_output_required=spatial, textual_output_required=True, temporal_requirement=temp_req
        ),
        is_compound=False,
        subtasks=[
            SubtaskSpec(
                subtask_id="subtask_1", description=query, primary_task=primary_task, target_entities=[],
                required_modalities=req_mods, temporal_requirement=temp_req,
                spatial_output_required=spatial, textual_output_required=True
            )
        ],
        observation_requirements=[
            ObservationRequirement(
                requirement_id="req_1", source_subtasks=[], minimum_observations=len(req_mods),
                maximum_observations=len(req_mods), required_modalities=req_mods, temporal_relationship=temp_req
            )
        ],
        ambiguous=False
    )


async def execute_agentic_pipeline(
    job_id: str,
    files_data: List[tuple],
    query: str,
    execution_mode: str = "real",
    qwen_backend: Optional[str] = None,
    planner_fallback: Optional[str] = None,
):
    """
    Runs the full Agentic Pipeline asynchronously.

    Flow:
        MC1 → Query Intelligence → MC3 (Select + Bind + Plan) → AgentController + Recovery
        → MC5 Evidence Normalization → Evidence Graph → MC6 Verification
        → Final Answer → MC8 Export
    """
    qwen_engine = None
    backend = None
    try:
        job = job_registry.get_job(job_id)
        caveats: List[str] = []
        planner_events: List[Dict] = []

        # ── STAGE 1: MC1 Input Qualification ──────────────────────────
        job_registry.update_status(job_id, "MC1_VALIDATING", {"message": "Validating inputs"})

        class MockUploadFile:
            def __init__(self, filename, content):
                self.filename = filename
                self._content = content
            async def read(self):
                return self._content
            async def seek(self, pos):
                pass

        upload_files = [MockUploadFile(f, b) for f, b in files_data]
        mc1_profile_legacy = await run_mc1_pipeline(upload_files, query)

        # Synthetic smoke-test profiles are a fixture-mode tool only; real jobs always use MC1's verdict.
        smoke_test = execution_mode == "fixture" and "smoke_test" in query.lower()
        if smoke_test:
            _fixture_smoke_profile(mc1_profile_legacy, query, files_data)

        job["mc1_profile"] = mc1_profile_legacy

        if not mc1_profile_legacy.get("task_executable"):
            _finish_early(
                job_id, "PRECONDITION_FAILED", "MC1 Preconditions failed",
                {
                    "final_answer": "The uploaded inputs did not pass input qualification. " + " ".join(mc1_profile_legacy.get("warnings") or []),
                    "execution_status": "PRECONDITION_FAILED",
                    "failed": [{"reason": w} for w in (mc1_profile_legacy.get("warnings") or [])],
                },
                {"details": mc1_profile_legacy.get("warnings")},
            )
            return

        # ── STAGE 1b: Load the uploaded pixels ────────────────────────
        rasters = {}
        if execution_mode == "real":
            from raster_io import save_uploads, load_job_rasters, to_model_tensor
            try:
                paths = save_uploads(job_id, files_data)
                rasters = load_job_rasters(paths, [f for f, _ in files_data])
            except Exception as e:
                _finish_early(
                    job_id, "PRECONDITION_FAILED", "Uploaded rasters could not be read",
                    {
                        "final_answer": f"The uploaded image pixels could not be read: {e}",
                        "execution_status": "PRECONDITION_FAILED",
                        "failed": [{"reason": str(e)}],
                    },
                )
                return
            tensors = {
                "t1": to_model_tensor(rasters["image_1"]) if "image_1" in rasters else None,
                "t2": to_model_tensor(rasters["image_2"]) if "image_2" in rasters else None,
            }
            job["rasters"] = rasters
        else:
            # Fixture mode: deterministic synthetic tensors, never presented as real imagery
            generator = torch.Generator().manual_seed(0)
            tensors = {
                "t1": torch.rand(1, 3, 256, 256, generator=generator),
                "t2": torch.rand(1, 3, 256, 256, generator=generator) if len(files_data) > 1 else None,
            }
        tensors["rasters"] = rasters
        tensors["job_id"] = job_id

        # ── STAGE 2: Build formal ObservationProfiles ─────────────────
        obs_list = _build_observation_profiles(mc1_profile_legacy)
        req_profile = RequestObservationProfile(
            observations=obs_list,
            spatial_overlap=mc1_profile_legacy.get("spatial_overlap"),
            coregistration_score=mc1_profile_legacy.get("coregistration_score"),
            compatibility=CompatibilityProfile(factors={"alignment": mc1_profile_legacy.get("alignment")}),
        )

        # ── STAGE 3: Query Intelligence ───────────────────────────────
        job_registry.update_status(job_id, "QUERY_INTELLIGENCE", {"message": "Understanding query intent"})

        backend = engine_factory.resolve_backend(qwen_backend)
        fallback_policy = (planner_fallback or os.getenv("SATQUERY_PLANNER_FALLBACK") or "registry").lower()
        planner_info = {"llm_backend": None, "query_intelligence": None, "tool_selection": None, "answer": None}

        if execution_mode == "real":
            if backend != "none":
                engine = engine_factory.create_engine(backend)
                try:
                    engine.load()
                    qwen_engine = engine
                    planner_info["llm_backend"] = engine_factory.model_label(backend)
                except Exception as e:
                    if fallback_policy == "strict":
                        _finish_early(
                            job_id, "MODEL_UNAVAILABLE", f"Qwen3 model ({backend}) cannot be loaded",
                            {
                                "final_answer": f"FINAL_ANSWER_MODEL_UNAVAILABLE: {str(e)}",
                                "execution_status": "MODEL_UNAVAILABLE",
                                "model_unavailable": [engine_factory.model_label(backend)],
                            },
                            {"error": str(e)},
                        )
                        return
                    caveats.append(f"Qwen3 ({engine_factory.model_label(backend)}) unavailable; the deterministic Tool Registry planner was used.")
                    job_registry.update_status(job_id, "QUERY_INTELLIGENCE", {
                        "message": "Qwen3 unavailable, falling back to Tool Registry rules",
                        "error": str(e),
                    })
        else:
            qwen_engine = mock_engine
            planner_info["llm_backend"] = "fixture mock"

        if smoke_test:
            qi_result = _fixture_smoke_intent(query)
            planner_info["query_intelligence"] = "fixture"
        else:
            qi_result = None
            if qwen_engine is not None:
                qi_result = QueryIntelligencePipeline(qwen_engine).process_query(query)
                planner_info["query_intelligence"] = "qwen3"
                if (
                    qi_result.ambiguous
                    or not qi_result.subtasks
                    or all(st.primary_task == "unknown" for st in qi_result.subtasks)
                ):
                    job_registry.update_status(job_id, "QUERY_INTELLIGENCE", {
                        "message": "Qwen3 could not resolve the intent; applying Tool Registry rules to the query and inputs",
                    })
                    qi_result = None
            if qi_result is None:
                qi_result = RegistryQueryInterpreter().process_query(query, mc1_profile_legacy)
                planner_info["query_intelligence"] = "registry_rules"

        if qi_result.ambiguous:
            _finish_early(
                job_id, "ABSTAIN", "Query intent could not be determined",
                {
                    "final_answer": (
                        "The request is ambiguous for the uploaded inputs. Please say what you want to know, "
                        "for example 'What changed between these images?' or 'Is there an airstrip in this image?'."
                    ),
                    "execution_status": "ABSTAIN",
                },
            )
            return

        job_registry.update_status(job_id, "QUERY_INTELLIGENCE", {
            "message": f"Intent: {qi_result.primary_task_spec.primary_task} (via {planner_info['query_intelligence']})",
            "task_spec": qi_result.primary_task_spec.model_dump(),
        })

        # ── STAGE 4: Tool Selection ───────────────────────────────────
        job_adapters = build_adapters(execution_mode)
        job_tool_registry, availability = build_job_registry(execution_mode, job_adapters)
        unavailable = [a for a in availability if not a["available"]]
        job_registry.update_status(job_id, "TOOL_SELECTION", {
            "message": "Selecting specialist capability from the Tool Registry",
            "details": [f"{a['tool_id']}: {'available' if a['available'] else 'unavailable - ' + a['reason']}" for a in availability],
        })

        qwen_selector = QwenToolSelector(qwen_engine, job_tool_registry) if qwen_engine is not None else None
        selector = FallbackToolSelector(qwen_selector, job_tool_registry, planner_events)
        try:
            calls = selector.select_tools(qi_result)
        except Exception as e:
            task = qi_result.primary_task_spec.primary_task
            blocked = [
                f"{a['name']}: {a['reason']}" for a in unavailable
                if (job_tool_registry.get(a["tool_id"]) and task in job_tool_registry.get(a["tool_id"]).supported_tasks)
            ]
            _finish_early(
                job_id, "ABSTAIN", "No capability can serve this request",
                {
                    "final_answer": (
                        f"No available specialist can perform '{task}' in this environment, so no answer is given. "
                        + (" ".join(blocked) if blocked else str(e))
                    ),
                    "failed": [{"reason": b} for b in blocked] or [{"reason": str(e)}],
                    "execution_status": "ABSTAIN",
                    "tool_registry": availability,
                },
            )
            return
        planner_info["tool_selection"] = planner_events[-1]["planner"] if planner_events else None
        # Engines need the semantic task (e.g. captioning vs. question answering), not just the query text
        subtask_tasks = {st.subtask_id: st.primary_task for st in qi_result.subtasks}
        for call in calls:
            task = next((subtask_tasks[s] for s in call.source_subtask_ids if s in subtask_tasks), None)
            call.arguments.setdefault("primary_task", task or qi_result.primary_task_spec.primary_task)
            call.arguments.setdefault("query", query)
        job_registry.update_status(job_id, "TOOL_SELECTION", {
            "message": f"Selected: {', '.join(c.tool_id for c in calls)} (via {planner_info['tool_selection']})",
            "details": [e["reason"] for e in planner_events if e.get("reason")],
        })

        # ── STAGE 5: Observation Binding ──────────────────────────────
        job_registry.update_status(job_id, "OBSERVATION_BINDING", {"message": "Binding physical observations"})

        if not qi_result.observation_requirements:
            raise ValueError("No observation requirements found in QueryIntelligenceResult")

        job_binder = ObservationBinder(assume_upload_order=True)
        bound_calls = []
        for call in calls:
            req = next((r for r in qi_result.observation_requirements if call.source_subtask_ids and call.source_subtask_ids[0] in r.source_subtasks), None)
            if not req:
                req = qi_result.observation_requirements[0]

            binding_result = job_binder.bind(call, req, req_profile)
            if binding_result.status != "SUFFICIENT":
                reasons = list(binding_result.failed_constraints)
                if binding_result.status == "AMBIGUOUS":
                    reasons.append(
                        f"Cannot decide which observation to use for {binding_result.unresolved_roles}; "
                        f"candidates: {binding_result.candidate_observation_ids}. Upload a single image or name the image in the query."
                    )
                _finish_early(
                    job_id, "INSUFFICIENT_OBSERVATIONS", f"Observation binding failed: {binding_result.status}",
                    {
                        "final_answer": f"Cannot execute: observations are {binding_result.status}. {' '.join(reasons)}",
                        "execution_status": "INSUFFICIENT_OBSERVATIONS",
                        "failed": [{"reason": r} for r in reasons],
                    },
                    {"failed_constraints": binding_result.failed_constraints, "warnings": binding_result.warnings},
                )
                return

            for w in binding_result.warnings:
                if "assumed" in w.lower() and w not in caveats:
                    caveats.append(w)
            bound_calls.append(binding_result.bound_call)

        # ── STAGE 6: Workflow Planning ────────────────────────────────
        job_registry.update_status(job_id, "WORKFLOW_PLANNING", {"message": "Building execution workflow"})
        plan = planner.build_workflow(bound_calls, qi_result, job_tool_registry)
        if plan.status != "ready":
            raise ValueError(f"Workflow Planning failed: {plan.errors}")

        # ── STAGE 7: Agentic Execution (with bounded recovery) ────────
        job_registry.update_status(job_id, "AGENTIC_EXECUTION", {"message": "Executing workflow with recovery bounds"})

        job_controller = AgentController(job_tool_registry, job_adapters)
        qwen_recovery = QwenRecoveryPlanner(qwen_engine) if qwen_engine is not None else None
        recovery_planner = FallbackRecoveryPlanner(qwen_recovery, job_tool_registry, planner_events)
        job_recovery_manager = RecoveryManager(job_controller, recovery_planner, job_tool_registry, planner)

        state = await asyncio.to_thread(
            job_recovery_manager.execute_with_recovery,
            plan, qi_result, mc1_profile_legacy, tensors, query
        )

        # ── STAGE 8: MC5 Evidence Normalization ───────────────────────
        job_registry.update_status(job_id, "EVIDENCE_NORMALIZATION", {"message": "Normalizing specialist outputs to evidence"})

        evidence = []
        model_unavailable_tools = []
        change_outputs = None

        for completed_call in state.completed_calls:
            result = state.results[completed_call]
            ev_list = [ev for ev in result.outputs.get("evidence", []) if isinstance(ev, dict) and "evidence_id" in ev]
            if not ev_list:
                caveats.append(f"{result.tool_id} completed but produced no evidence objects.")
            evidence.extend(ev_list)
            if result.outputs.get("change_statistics") is not None:
                change_outputs = result.outputs

        for failed_call in state.failed_calls:
            result = state.results.get(failed_call)
            if result and result.error_information and "model_unavailable" in result.error_information.lower():
                model_unavailable_tools.append(result.tool_id)

        # ── STAGE 9: Evidence Graph ───────────────────────────────────
        graph = EvidenceGraph()
        query_node_id = f"query_{job_id}"
        graph.add_node(query_node_id, "query", {"text": query})
        for ev in evidence:
            graph.insert_evidence(ev, query_node_id)

        job["evidence_objects"] = evidence
        job["evidence_graph"] = graph.to_dict()

        # ── STAGE 10: MC6 Verification Gate ───────────────────────────
        job_registry.update_status(job_id, "VERIFICATION", {"message": "Verifying evidence claims"})
        job_verifier = Verifier()
        raw_verification = job_verifier.verify(evidence, mc1_profile_legacy)
        verified, rejected, global_triggers = _partition_evidence(evidence, raw_verification)
        if evidence and not verified:
            verification_result = {**raw_verification, "status": "INSUFFICIENT_EVIDENCE", "rejected_claims": rejected}
        elif not evidence:
            verification_result = raw_verification
        else:
            verification_result = {**raw_verification, "status": "VERIFIED", "rejected_claims": rejected}
        for trig in global_triggers:
            if isinstance(trig, dict):
                caveats.append(f"Verification warning: {trig.get('trigger')} ({ {k: v for k, v in trig.items() if k != 'trigger'} })")
        job_registry.update_status(job_id, "VERIFICATION", {
            "message": f"{len(verified)} evidence item(s) verified, {len(rejected)} rejected",
            "details": [f"{r.get('evidence_id')}: rejected (low confidence {r.get('confidence')})" for r in rejected],
        })

        # ── STAGE 11: Final Answer Synthesis ──────────────────────────
        job_registry.update_status(job_id, "ANSWER_SYNTHESIS", {"message": "Synthesizing final grounded answer"})

        limitations = list(caveats)
        if change_outputs is not None:
            limitations.append(
                "The change detector localises where the radar/optical signal changed; it does not identify what the "
                "changed objects are (e.g. it cannot by itself confirm an airstrip or buildings)."
            )
        for completed_call in state.completed_calls:
            outputs = state.results[completed_call].outputs
            if outputs.get("domain_mismatch_flag"):
                obs_id = outputs.get("observation_id", "image_1")
                img = mc1_profile_legacy.get(obs_id) or {}
                warning = (
                    f"{img.get('filename', obs_id)} is not an optical image (detected modality: {img.get('modality', 'unknown')}). "
                    "PaliGemma is an optical vision-language model, so its answer on this image is unreliable; "
                    "its confidence was multiplied by 0.3."
                )
                limitations.append(warning)
                if warning not in caveats:
                    caveats.append(warning)

        final_answer = _synthesize_final_answer(
            query=query,
            evidence=verified,
            verification_result=verification_result,
            agent_state_status=state.status,
            model_unavailable_tools=model_unavailable_tools,
            execution_mode=execution_mode,
            qwen_engine=qwen_engine,
            limitations=limitations,
        )
        planner_info["answer"] = final_answer.get("answer_generator")

        # Attach agent execution metadata
        if state.replan_count > 0:
            caveats.append(f"Recoveries triggered: {state.replan_count}")
        if model_unavailable_tools:
            caveats.append(f"Unavailable models: {', '.join(model_unavailable_tools)}")
        if change_outputs is not None:
            caveats.append(limitations[-1])
            caveats.extend(change_outputs.get("notes", []))
        if rejected:
            caveats.append(f"{len(rejected)} evidence item(s) rejected by verification and excluded from the answer.")
        final_answer["caveats"] = caveats
        final_answer["confidence_note"] = "Highest uncalibrated evidence confidence; see each evidence item for its source."
        final_answer["planner"] = planner_info
        final_answer["tool_registry"] = availability
        final_answer["observations"] = {
            obs_id: {
                "filename": r.filename,
                "bands": int(r.data.shape[0]),
                "width": r.width,
                "height": r.height,
                "georeferenced": r.is_georeferenced,
                "bounds_wgs84": image_bounds_wgs84(r),
                "preview_url": f"/api/jobs/{job_id}/preview/{obs_id}",
            }
            for obs_id, r in rasters.items()
        }
        final_answer["change_statistics"] = change_outputs.get("change_statistics") if change_outputs else None
        final_answer["change_overlay"] = (
            {
                "url": f"/api/jobs/{job_id}/export/png",
                "bounds_wgs84": change_outputs.get("overlay_bounds"),
                "before_observation": change_outputs.get("before_observation"),
                "after_observation": change_outputs.get("after_observation"),
            } if change_outputs else None
        )
        final_answer["change_map"] = None
        final_answer["agent_state"] = state.model_dump()
        final_answer["verification_result"] = verification_result

        job["result"] = final_answer
        if change_outputs is not None:
            job["change_overlay_png"] = change_outputs.get("overlay_png")

        # ── STAGE 12: Finalize ────────────────────────────────────────
        if final_answer["execution_status"] == "INSUFFICIENT_EVIDENCE":
            final_status = "INSUFFICIENT_EVIDENCE"
        else:
            final_status = "DONE"

        job_registry.update_status(job_id, final_status, {"message": "Agentic pipeline complete"})

        from mc8_export.exporter import generate_exports
        job["exports"] = generate_exports(job)

    except Exception as e:
        job = job_registry.get_job(job_id)
        job_registry.update_status(job_id, "FAILED", {
            "message": "Internal execution error",
            "error": str(e),
            "traceback": traceback.format_exc()
        })
        job["result"] = job.get("result") or {
            "final_answer": f"Internal execution error: {e}",
            "execution_status": "FAILED",
            "claims": [], "evidence_references": [], "confidence": 0.0,
        }

        try:
            from mc8_export.exporter import generate_exports
            job["exports"] = generate_exports(job)
        except Exception:
            pass

    finally:
        # Release the job-scoped LLM (Ollama keeps its own cache; transformers weights are freed)
        if execution_mode == "real" and qwen_engine is not None and hasattr(qwen_engine, 'unload') and backend == "transformers":
            qwen_engine.unload()
