import asyncio
import sys
import os
import json
from unittest.mock import MagicMock

from mc1.pipeline import run_mc1_pipeline
from mc3_planner.workflow_planner import build_workflow_plan
from mc3_planner.tool_registry import ToolRegistry
from mc4b_temporal.tool_adapter import CHANGE_MAMBA_TOOL
from mc4a_vqa.tool_entry import PALIGEMMA_VQA_TOOL

class MockUploadFile:
    def __init__(self, filename, content):
        self.filename = filename
        self.content = content
        
    async def read(self):
        return self.content
        
    async def seek(self, pos):
        pass

async def test_mc23(filepaths, query):
    files = []
    for fp in filepaths:
        with open(fp, "rb") as f:
            content = f.read()
        files.append(MockUploadFile(os.path.basename(fp), content))
        
    mc1_profile = await run_mc1_pipeline(files, query)
    
    # Run the MC2 logic from job_manager.py
    q_lower = query.lower()
    temporal_req = "bi_temporal_change" if any(k in q_lower for k in ["changed", "change", "increase", "decrease", "expand"]) else ""
    
    mods = []
    if mc1_profile.get("image_1", {}).get("modality"):
        mods.append(mc1_profile["image_1"]["modality"])
    if mc1_profile.get("image_2", {}).get("modality"):
        mods.append(mc1_profile["image_2"]["modality"])
        
    required_mods = ["optical"] # fallback if empty
    if mods:
        required_mods = list(set(mods))
    
    task_spec = {
        "primary_task": "change_detection" if temporal_req else "single_image_vqa",
        "target_entities": ["built-up area"],
        "required_operations": ["temporal_analysis", "spatial_localization"] if temporal_req else ["visual_question_answering"],
        "required_modalities": required_mods,
        "temporal_requirement": temporal_req,
        "spatial_output_required": bool(temporal_req),
        "textual_output_required": True
    }
    
    print("MC2 Task Spec:")
    print(json.dumps(task_spec, indent=2))
    
    # Run MC3
    registry = ToolRegistry()
    registry.register(CHANGE_MAMBA_TOOL)
    registry.register(PALIGEMMA_VQA_TOOL)
    try:
        plan = build_workflow_plan(task_spec, mc1_profile, registry)
        print("\nMC3 Plan:")
        print(json.dumps(plan, indent=2))
    except Exception as e:
        print("\nMC3 Plan Failed:", str(e))


if __name__ == "__main__":
    filepaths = sys.argv[1:-1]
    query = sys.argv[-1]
    asyncio.run(test_mc23(filepaths, query))
