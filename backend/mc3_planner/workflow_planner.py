"""
workflow_planner.py — MC3.2 Workflow Planner.

Builds an Executable Workflow Plan by:
1. Matching MC2's task specification against the Tool Registry
2. Constraint-filtering candidates using the MC1 profile (fail fast at planning time)
3. Assembling the execution order with tool parameters
"""

from typing import Dict, List, Optional
from .tool_registry import ToolRegistry


def build_workflow_plan(
    task_spec: Dict,
    mc1_profile: Dict,
    registry: ToolRegistry,
) -> Dict:
    """
    Build an Executable Workflow Plan from an MC2 task specification.

    Args:
        task_spec: MC2 Structured Task Specification.
        mc1_profile: MC1 Structured Input Profile.
        registry: MC3.1 Tool Registry.

    Returns:
        Workflow Plan dict matching the MC3.2 schema, or a rejection dict.
    """
    temporal_req = task_spec.get("temporal_requirement", "")
    required_ops = task_spec.get("required_operations", [])
    required_modalities = task_spec.get("required_modalities", [])

    # 1. Find candidate tools by temporal requirement
    candidates = []
    if temporal_req:
        candidates = registry.find_by_temporal_requirement(temporal_req)

    # If no temporal match, try by primary task capability
    if not candidates:
        primary_task = task_spec.get("primary_task", "")
        candidates = registry.find_by_capability(primary_task)

    if not candidates:
        return {
            "status": "NO_TOOL_FOUND",
            "reason": (
                f"No tool found for temporal_requirement='{temporal_req}', "
                f"primary_task='{task_spec.get('primary_task')}'"
            ),
        }

    # 2. Constraint-filter candidates against MC1 profile (fail fast)
    viable = []
    for tool in candidates:
        constraints = tool.get("input_constraints", {})

        # Check image count
        required_count = constraints.get("image_count")
        
        # If image_count is 1 but we have 2 images, check if we can resolve the target image
        if required_count == 1 and mc1_profile.get("image_count") == 2:
            # Heuristic scan for ordinal cues
            query = task_spec.get("query", "").lower()
            ordinal_cues = ["first", "second", "image 1", "image 2", "optical one"]
            if any(cue in query for cue in ordinal_cues):
                pass # Can resolve, let it pass
            else:
                pass # Ambiguous, but we let it pass and set ambiguity_flag later
        elif required_count and mc1_profile.get("image_count") != required_count:
            continue

        # Check modality support
        supported_mods = tool.get("supported_modalities", [])
        if required_modalities:
            if not all(m in supported_mods for m in required_modalities):
                continue

        # Check spatial overlap threshold
        min_overlap = constraints.get("min_spatial_overlap", 0)
        actual_overlap = mc1_profile.get("spatial_overlap", 0) or 0
        if actual_overlap < min_overlap:
            continue

        # Check coregistration
        min_coreg = constraints.get("min_coregistration_score", 0)
        actual_coreg = mc1_profile.get("coregistration_score", 0) or 0
        if actual_coreg < min_coreg:
            continue

        # Check GSD range
        gsd_range = tool.get("spatial_resolution_range", {})
        img1_gsd = mc1_profile.get("image_1", {}).get("gsd_m")
        if img1_gsd:
            min_gsd = gsd_range.get("min_gsd_m", 0)
            max_gsd = gsd_range.get("max_gsd_m", float("inf"))
            if not (min_gsd <= img1_gsd <= max_gsd):
                continue

        viable.append(tool)

    if not viable:
        # Check specifically for SAR+SAR change detection failure
        if task_spec.get("primary_task") == "change_detection" and "sar" in required_modalities:
            return {
                "status": "PRECONDITION_FAILED",
                "reason": "no registered tool accepts sar+sar for change detection",
                "failed": [{"check": "supported_modalities", "reason": "No registered tool accepts sar+sar for change detection", "actual": "sar"}]
            }
            
        return {
            "status": "CONSTRAINTS_NOT_MET",
            "reason": "No tool passed constraint filtering against MC1 profile.",
            "candidates_checked": len(candidates),
        }

    # 3. Select the best tool (for now: first viable candidate)
    selected = viable[0]

    # 4. Determine sub-task from task spec
    sub_task = "binary_change_detection"
    if "semantic" in str(required_ops):
        sub_task = "semantic_change_detection"

    # 5. Build the Executable Workflow Plan
    tool_name = selected["name"]
    
    if tool_name == "PALIGEMMA_VQA":
        plan = {
            "status": "PLAN_READY",
            "selected_tools": [tool_name],
            "execution_order": [tool_name],
            "tool_parameters": {
                tool_name: {
                    "query": task_spec.get("query", ""),
                    "target_entities": task_spec.get("target_entities", []),
                },
            },
            "expected_outputs": [
                "textual_answer",
                "model_confidences",
                "spatial_evidence",
            ],
            "verification_requirements": [
                "cross_modal_corroboration_if_conflict",
            ],
        }
    else:
        plan = {
            "status": "PLAN_READY",
            "selected_tools": [tool_name],
            "execution_order": [tool_name],
            "tool_parameters": {
                tool_name: {
                    "sub_task": sub_task,
                    "query": task_spec.get("query", ""),
                    "target_entities": task_spec.get("target_entities", []),
                },
            },
            "expected_outputs": [
                "change_mask",
                "change_regions",
                "change_statistics",
            ],
            "verification_requirements": [
                "cross_modal_corroboration_if_conflict",
            ],
        }

    return plan
