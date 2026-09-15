import json
from typing import List, Dict, Any
from .schemas import TaskSpec, SubtaskSpec, DecompositionResult
from .inference import Qwen3Inference

class QueryDecomposer:
    """
    Decomposes a query into atomic analytical subtasks using Qwen3.
    """
    
    def __init__(self, inference_engine: Qwen3Inference):
        self.inference_engine = inference_engine
        
    def _build_prompt(self, query: str, primary_task_spec: TaskSpec) -> str:
        return (
            "You are a query decomposition engine for a remote sensing AI system.\n"
            "Your job is to determine if the user query contains multiple analytical requirements (COMPOUND) or just one (ATOMIC).\n"
            "If compound, break it down into a list of atomic semantic subtasks.\n"
            "Do NOT include any surrounding text, markdown blocks, or explanations. Output ONLY valid JSON.\n\n"
            "The JSON must strictly follow this schema:\n"
            "{\n"
            '  "is_compound": boolean,\n'
            '  "ambiguous": boolean (true if intent is unclear or lacks detail to decompose),\n'
            '  "subtasks": [\n'
            '    {\n'
            '      "subtask_id": "string identifier, e.g., subtask_1",\n'
            '      "description": "Natural language description of this subtask",\n'
            '      "primary_task": "one of [single_image_vqa, captioning, grounding, change_detection, change_vqa, cross_modal_fusion, unknown]",\n'
            '      "target_entities": ["list", "of", "objects"],\n'
            '      "required_modalities": ["list containing optical, sar, or unspecified"],\n'
            '      "temporal_requirement": "one of [before_after, none, unknown]",\n'
            '      "spatial_output_required": boolean,\n'
            '      "textual_output_required": boolean,\n'
            '      "depends_on": ["list", "of", "subtask_ids", "this", "depends", "on"]\n'
            '    }\n'
            '  ]\n'
            "}\n\n"
            "Decomposition Rules:\n"
            "1. ATOMIC: If the query only has one analytical goal (e.g. 'What changed?', 'Where are buildings?', 'Compare SAR and optical'), set is_compound=false and output EXACTLY ONE subtask matching the primary intent.\n"
            "2. COMPOUND: If the query asks for multiple distinct things (e.g., 'Identify changes AND locate them', 'Describe scene AND find roads'), set is_compound=true and output multiple subtasks.\n"
            "3. PRESERVE CONTEXT: If the overall query is about change over time, propagate temporal_requirement='before_after' to all relevant subtasks. If it asks about specific entities, propagate them.\n"
            "4. DEPENDENCIES: Use depends_on to link subtasks semantically (e.g., 'locate them' depends on 'find changes').\n"
            "5. NO OVER-DECOMPOSITION: Do not split 'Describe what changed' into 3 steps. That is ONE atomic change_detection task.\n"
            "6. NO UNDER-DECOMPOSITION: Do not drop requirements. If they want localization (where), ensure a grounding subtask exists.\n"
            "7. CRITICAL: DO NOT output tool names, model names, specialist names, or execution functions (e.g., no ChangeMamba, PaliGemma). Focus ONLY on semantic intent (WHAT, not HOW).\n\n"
            f"Query: \"{query}\"\n"
            f"Primary Detected Intent: {primary_task_spec.primary_task}\n\n"
            "JSON Output:\n"
        )
        
    def _parse_json(self, response_text: str) -> dict:
        """Safely extract and parse JSON from the LLM output."""
        text = response_text.strip()
        
        # Strip markdown if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
            
        if text.endswith("```"):
            text = text[:-3]
            
        text = text.strip()
        
        start = text.find("{")
        end = text.rfind("}")
        
        if start != -1 and end != -1 and end >= start:
            text = text[start:end+1]
            
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
            
    def _create_fallback_subtask(self, primary_task_spec: TaskSpec) -> SubtaskSpec:
        """Creates a single atomic subtask from the primary TaskSpec."""
        return SubtaskSpec(
            subtask_id="subtask_1",
            description=f"Perform {primary_task_spec.primary_task}",
            primary_task=primary_task_spec.primary_task,
            target_entities=primary_task_spec.target_entities,
            required_modalities=primary_task_spec.required_modalities,
            temporal_requirement=primary_task_spec.temporal_requirement,
            spatial_output_required=primary_task_spec.spatial_output_required,
            textual_output_required=primary_task_spec.textual_output_required,
            depends_on=[],
            ambiguous=primary_task_spec.ambiguous
        )

    def decompose(self, query: str, primary_task_spec: TaskSpec) -> DecompositionResult:
        """
        Runs the decomposition prompt and validates the resulting DecompositionResult.
        """
        fallback_result = DecompositionResult(
            original_query=query,
            primary_task_spec=primary_task_spec,
            is_compound=False,
            subtasks=[self._create_fallback_subtask(primary_task_spec)],
            ambiguous=True
        )
        
        prompt = self._build_prompt(query, primary_task_spec)
        
        result = self.inference_engine.generate(prompt, max_new_tokens=1024, temperature=0.01)
        
        if result["status"] != "ok" or not result["response"]:
            return fallback_result
            
        parsed_dict = self._parse_json(result["response"])
        
        if not parsed_dict:
            return fallback_result
            
        try:
            # Validate the subtasks list
            raw_subtasks = parsed_dict.get("subtasks", [])
            subtasks = [SubtaskSpec(**st) for st in raw_subtasks]
            
            # If model didn't output subtasks for some reason, fallback
            if not subtasks:
                return fallback_result
                
            return DecompositionResult(
                original_query=query,
                primary_task_spec=primary_task_spec,
                is_compound=parsed_dict.get("is_compound", False),
                subtasks=subtasks,
                ambiguous=parsed_dict.get("ambiguous", False)
            )
        except Exception:
            # If Pydantic fails (e.g., hallucinated primary_task or bad depends_on types), we fallback safely
            return fallback_result
