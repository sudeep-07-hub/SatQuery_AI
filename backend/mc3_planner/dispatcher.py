"""
dispatcher.py — MC3 Plan Executor.

Executes an Executable Workflow Plan by calling the appropriate tool adapters
in the specified execution order.
"""

import torch
from typing import Dict, Optional

from mc4b_temporal.tool_adapter import ChangeMambaAdapter
from mc4b_temporal.evidence_normalizer import normalize_to_evidence
from mc4a_vqa.specialist import PaliGemmaVQASpecialist
from mc4a_vqa.evidence_normalizer import to_evidence_object

# Adapter registry (lazy-initialized)
_adapters = {}


def _get_adapter(tool_name: str):
    """Lazy-initialize tool adapters."""
    if tool_name not in _adapters:
        if tool_name == "CHANGE_MAMBA":
            _adapters[tool_name] = ChangeMambaAdapter()
        elif tool_name == "PALIGEMMA_VQA":
            _adapters[tool_name] = PaliGemmaVQASpecialist()
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

        if tool_name == "PALIGEMMA_VQA":
            # Determine target image
            target_image_key = "image_1"
            guessed_target = False
            
            if mc1_profile.get("image_count") == 2 and "ambiguity_flag" not in mc1_profile:
                query = params.get("query", "").lower()
                if "second" in query or "image 2" in query:
                    target_image_key = "image_2"
                elif "first" not in query and "image 1" not in query:
                    guessed_target = True
            
            target_tensor = t1 if target_image_key == "image_1" else t2
            if target_tensor is None:
                return {
                    "status": "TOOL_FAILED",
                    "reason": f"Missing image tensor for {tool_name}",
                }
                
            # Convert (C, H, W) or (B, C, H, W) tensor to PIL Image without torchvision
            import numpy as np
            from PIL import Image
            
            # Ensure it's on CPU, detach, and convert to numpy
            arr = target_tensor.cpu().detach().numpy()
            
            # Remove batch dimension if present
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr[0]
            
            if arr.ndim == 3:
                if arr.shape[0] in [1, 3]:
                    # Convert (C, H, W) to (H, W, C)
                    arr = np.transpose(arr, (1, 2, 0))
                # Handle single band
                if arr.shape[2] == 1:
                    arr = arr[:, :, 0]
                    
            # Normalize to 0-255 uint8 if it's float
            if arr.dtype.kind == 'f':
                # Assuming typical 0-1 range if float, or fallback to min/max scaling
                arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
                
            pil_image = Image.fromarray(arr)
            
            # Execute VQA
            result = adapter.run(
                image_input=pil_image,
                query=params.get("query", ""),
                mc1_profile=mc1_profile,
                mc2_task_spec=params
            )
            
            if "blocked_reason" in result:
                return {
                    "status": "PRECONDITION_FAILED",
                    "tool": tool_name,
                    "failed": [{"check": "quality_gate", "reason": result["blocked_reason"]}],
                }
                
            if guessed_target:
                result["warnings"] = ["Ambiguous query against multi-image input. Defaulted to evaluating image_1."]
                
            tool_outputs[tool_name] = result
            
            # Normalize to Evidence Objects (MC5.1)
            evidence = to_evidence_object(result, mc1_profile, params)
            evidence_objects.extend(evidence)
            
        elif tool_name == "CHANGE_MAMBA":
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
