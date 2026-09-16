"""
rule_based.py — Deterministic, registry-driven planning used when Qwen3 is unavailable
(or returns unusable output).

Nothing here imitates an LLM: query intent comes from explicit keyword rules plus the
shape of the uploaded inputs, and engines are chosen from the Tool Registry by declared
capability and availability. Every result produced this way is labelled
`planner = "registry_rules"` in the job trace.
"""
import re
import uuid
from typing import Dict, List, Optional

from agent.registry import AgentToolRegistry
from agent.schemas import FailureContext, InputBinding, ReplanRequest, ToolCall, ToolSpec
from qwen.pipeline import QueryIntelligenceResult
from qwen.requirements import ObservationRequirementGenerator
from qwen.schemas import DecompositionResult, SubtaskSpec, TaskSpec

CHANGE_WORDS = (
    "change", "changed", "changes", "differ", "difference", "before", "after", "increase", "decrease",
    "new ", "appear", "disappear", "removed", "cleared", "clearing", "deforest", "built", "construction",
    "expand", "expansion", "grown", "growth", "lost", "loss", "compare", "comparison", "between",
)
FUSION_WORDS = ("fuse", "fusion", "combine", "both sar and optical", "sar and optical", "optical and sar", "cross-modal", "cross modal", "same scene", "consistent")
CAPTION_WORDS = ("describe", "description", "caption", "summarize", "summarise", "overview", "what is in", "what's in")
GROUNDING_WORDS = ("where is", "where are", "locate", "highlight", "point to", "find the")
YES_NO_START = re.compile(r"^\s*(is|are|was|were|did|does|do|has|have|can|could)\b", re.IGNORECASE)


def _modalities(mc1_profile: Dict) -> List[str]:
    mods = []
    for key in ("image_1", "image_2"):
        img = mc1_profile.get(key)
        if img:
            mods.append(img.get("modality", "unknown"))
    return mods


def classify_query(query: str, mc1_profile: Dict) -> Optional[str]:
    """Returns a Phase 2 task type, or None when the intent cannot be decided safely."""
    q = f" {query.lower().strip()} "
    mods = _modalities(mc1_profile)
    count = len(mods)

    if count == 2:
        if set(mods) == {"optical", "sar"} or any(w in q for w in FUSION_WORDS):
            return "cross_modal_fusion"
        if any(w in q for w in CHANGE_WORDS):
            return "change_vqa" if YES_NO_START.match(query) else "change_detection"
        return None
    if count == 1:
        if any(w in q for w in CAPTION_WORDS):
            return "captioning"
        if any(w in q for w in GROUNDING_WORDS):
            return "grounding"
        return "single_image_vqa"
    return None


class RegistryQueryInterpreter:
    """Deterministic replacement for the Qwen3 QueryIntelligencePipeline."""

    def process_query(self, query: str, mc1_profile: Dict) -> QueryIntelligenceResult:
        task = classify_query(query, mc1_profile)
        ambiguous = task is None
        task = task or "unknown"
        temporal = "before_after" if task in ("change_detection", "change_vqa") else "none"
        required_modalities = ["optical", "sar"] if task == "cross_modal_fusion" else ["unspecified"]

        spec = TaskSpec(
            query=query,
            primary_task=task,
            required_modalities=required_modalities,
            temporal_requirement=temporal,
            spatial_output_required=task in ("change_detection", "change_vqa", "grounding", "cross_modal_fusion"),
            textual_output_required=True,
            ambiguous=ambiguous,
        )
        subtask = SubtaskSpec(
            subtask_id="subtask_1",
            description=query,
            primary_task=task,
            required_modalities=required_modalities,
            temporal_requirement=temporal,
            spatial_output_required=spec.spatial_output_required,
            textual_output_required=True,
            ambiguous=ambiguous,
        )
        decomposition = DecompositionResult(
            original_query=query, primary_task_spec=spec, is_compound=False, subtasks=[subtask], ambiguous=ambiguous
        )
        return QueryIntelligenceResult(
            original_query=query,
            primary_task_spec=spec,
            is_compound=False,
            subtasks=[subtask],
            observation_requirements=ObservationRequirementGenerator().generate(decomposition),
            ambiguous=ambiguous,
        )


_STATUS_RANK = {"implemented": 0, "partial": 1, "mocked": 2, "disconnected": 3, "unavailable": 4}


def rank_tools(tools: List[ToolSpec]) -> List[ToolSpec]:
    return sorted(tools, key=lambda t: (_STATUS_RANK.get(t.status, 9), t.tool_id))


class RegistryToolSelector:
    """Chooses, for each subtask, the best enabled registry capability that supports its task."""

    def __init__(self, registry: AgentToolRegistry):
        self.registry = registry

    def select_tools(self, intent: QueryIntelligenceResult) -> List[ToolCall]:
        if intent.ambiguous:
            raise ValueError("Query intent is ambiguous. Safe fallback: Tool selection rejected.")
        calls = []
        for subtask in intent.subtasks:
            candidates = rank_tools([t for t in self.registry.find_by_task(subtask.primary_task) if t.enabled])
            if not candidates:
                raise ValueError(f"No enabled capability in the registry supports task '{subtask.primary_task}'")
            req = next((r for r in intent.observation_requirements if subtask.subtask_id in r.source_subtasks), None)
            req = req or (intent.observation_requirements[0] if intent.observation_requirements else None)
            bindings = {"primary": InputBinding(requirement_id=req.requirement_id)} if req else {}
            calls.append(ToolCall(
                call_id=f"call_{uuid.uuid4().hex[:8]}",
                tool_id=candidates[0].tool_id,
                arguments={"query": intent.original_query},
                source_subtask_ids=[subtask.subtask_id],
                input_bindings=bindings,
            ))
        return calls


class RegistryRecoveryPlanner:
    """On failure, proposes another enabled capability that supports the same task(s)."""

    def __init__(self, registry: AgentToolRegistry):
        self.registry = registry

    def propose_recovery(self, context: FailureContext, available_tools: List[ToolSpec], query: str) -> ReplanRequest:
        failed = self.registry.get(context.failed_tool_id)
        failed_tasks = set(failed.supported_tasks) if failed else set()
        alternatives = rank_tools([
            t for t in available_tools
            if t.tool_id != context.failed_tool_id and failed_tasks & set(t.supported_tasks)
        ])
        if not alternatives:
            return ReplanRequest(
                recovery_possible=False,
                reason=f"No other registered capability supports {sorted(failed_tasks) or 'this task'}.",
            )
        return ReplanRequest(
            recovery_possible=True,
            reason=f"Registry rule: '{alternatives[0].tool_id}' supports the same task as the failed '{context.failed_tool_id}'.",
            replacement_tool_id=alternatives[0].tool_id,
            arguments={"query": query},
        )


class FallbackToolSelector:
    """Qwen3 selection first; deterministic registry selection if Qwen3 fails or proposes an invalid tool."""

    def __init__(self, primary, registry: AgentToolRegistry, events: List[Dict]):
        self.primary = primary
        self.fallback = RegistryToolSelector(registry)
        self.events = events

    def select_tools(self, intent: QueryIntelligenceResult) -> List[ToolCall]:
        if self.primary is not None:
            try:
                calls = self.primary.select_tools(intent)
                self.events.append({"component": "tool_selection", "planner": "qwen3"})
                return calls
            except Exception as e:
                self.events.append({"component": "tool_selection", "planner": "registry_rules", "reason": f"Qwen3 selection rejected: {e}"})
        else:
            self.events.append({"component": "tool_selection", "planner": "registry_rules"})
        return self.fallback.select_tools(intent)


class FallbackRecoveryPlanner:
    def __init__(self, primary, registry: AgentToolRegistry, events: List[Dict]):
        self.primary = primary
        self.fallback = RegistryRecoveryPlanner(registry)
        self.events = events

    def propose_recovery(self, context: FailureContext, available_tools: List[ToolSpec], query: str) -> ReplanRequest:
        if self.primary is not None:
            try:
                plan = self.primary.propose_recovery(context, available_tools, query)
                self.events.append({"component": "recovery", "planner": "qwen3"})
                return plan
            except Exception as e:
                self.events.append({"component": "recovery", "planner": "registry_rules", "reason": f"Qwen3 recovery rejected: {e}"})
        return self.fallback.propose_recovery(context, available_tools, query)


def template_answer(query: str, evidence: List[Dict], rejected: List[Dict]) -> str:
    """Evidence-only answer used without an LLM: restates verified claims, adds nothing."""
    if not evidence:
        return "No evidence was produced, so the query cannot be answered."
    parts = [ev.get("claim", "") for ev in evidence if ev.get("claim")]
    answer = " ".join(parts)
    if rejected:
        answer += f" ({len(rejected)} claim(s) were rejected by verification and are not included.)"
    return answer
