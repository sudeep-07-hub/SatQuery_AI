"""
dispatcher.py — MC3 Plan Executor.

Executes an Executable Workflow Plan by calling the appropriate tool adapters
in the specified execution order.
"""

import torch
from typing import Dict, Optional

from mc4b_temporal.tool_adapter import ChangeMambaAdapter
from mc4b_temporal.evidence_normalizer import normalize_to_evidence


# Adapter registry (lazy-initialized)
_adapters = {}


def _get_adapter(tool_name: str):
    """Lazy-initialize tool adapters."""
    if tool_name not in _adapters:
        if tool_name == "CHANGE_MAMBA":
            _adapters[tool_name] = ChangeMambaAdapter()
        else:
            raise ValueError(f"No adapter for tool: {tool_name}")
    return _adapters[tool_name]


def execute_plan(
    plan: Dict,
    mc1_profile: Dict,
    image_tensors: Dict[str, torch.Tensor],
    query: str,
    seed: Optional[int] = None,
) -> Dict:
    """
    Execute a Workflow Plan.

    Args:
        plan: Output of build_workflow_plan().
        mc1_profile: MC1 Structured Input Profile.
        image_tensors: {"t1": tensor, "t2": tensor}
        query: Original user query.
        seed: Optional seed for reproducibility.

    Returns:
        dict with:
            tool_outputs: {tool_name: raw_output}
            evidence_objects: list of MC5.1 Evidence Objects
            status: "SUCCESS" | "TOOL_FAILED" | "PRECONDITION_FAILED"
    """
    if plan.get("status") != "PLAN_READY":
        return {
            "status": "PLAN_INVALID",
            "reason": plan.get("reason", "Plan is not executable"),
        }

    tool_outputs = {}
    evidence_objects = []

    for tool_name in plan["execution_order"]:
        adapter = _get_adapter(tool_name)
        params = plan.get("tool_parameters", {}).get(tool_name, {})

        t1 = image_tensors.get("t1")
        t2 = image_tensors.get("t2")

        if t1 is None or t2 is None:
            return {
                "status": "TOOL_FAILED",
                "reason": f"Missing image tensors for {tool_name}",
            }

        # Execute the tool
        result = adapter.execute(mc1_profile, t1, t2, query=query, seed=seed)

        if result.get("status") == "PRECONDITION_FAILED":
            return {
                "status": "PRECONDITION_FAILED",
                "tool": tool_name,
                "failed": result.get("failed", []),
            }

        tool_outputs[tool_name] = result

        # Normalize to Evidence Objects (MC5.1)
        evidence = normalize_to_evidence(result, mc1_profile, query)
        evidence_objects.extend(evidence)

    return {
        "status": "SUCCESS",
        "tool_outputs": tool_outputs,
        "evidence_objects": evidence_objects,
    }
