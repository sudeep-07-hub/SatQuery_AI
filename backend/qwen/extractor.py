import json
from .schemas import TaskSpec
from .inference import Qwen3Inference

class TaskSpecExtractor:
    """
    Extracts structured TaskSpec intent from a natural language query using Qwen3.
    """
    
    def __init__(self, inference_engine: Qwen3Inference):
        self.inference_engine = inference_engine
        
    def _build_prompt(self, query: str) -> str:
        return (
            "You are an intent extraction engine for a remote sensing AI system.\n"
            "Your ONLY job is to extract the user's analytical intent and output valid JSON.\n"
            "Do NOT include any surrounding text, markdown blocks, or explanations. Output ONLY JSON.\n\n"
            "The JSON must strictly follow this schema:\n"
            "{\n"
            '  "query": "The original user query",\n'
            '  "primary_task": "one of [single_image_vqa, captioning, grounding, change_detection, change_vqa, cross_modal_fusion, unknown]",\n'
            '  "target_entities": ["list", "of", "objects", "or", "features"],\n'
            '  "required_modalities": ["list containing optical, sar, or unspecified"],\n'
            '  "temporal_requirement": "one of [before_after, none, unknown]",\n'
            '  "spatial_output_required": boolean (true if locating/grounding/detecting boundaries),\n'
            '  "textual_output_required": boolean,\n'
            '  "ambiguous": boolean (true if the query lacks necessary detail or is confusing)\n'
            "}\n\n"
            "Semantic Classification Rules (Evaluate in order of precedence):\n"
            "1. If they ask a YES/NO or specific question about change over time (e.g., 'Did buildings increase?'), use change_vqa AND temporal_requirement=before_after.\n"
            "2. If they ask to identify, describe, or characterize changes (e.g., 'Describe the changes', 'What changed?'), use change_detection AND temporal_requirement=before_after.\n"
            "3. If they ask to locate, find, or show WHERE something is (e.g., 'Where are the buildings?'), use grounding AND spatial_output_required=true.\n"
            "4. If they explicitly ask to compare or use BOTH SAR and optical, use cross_modal_fusion AND required_modalities=[\"optical\", \"sar\"].\n"
            "5. If they ask for a general scene description or summary (e.g., 'Describe this image'), use captioning.\n"
            "6. If they ask a specific question about a single image (e.g., 'Are there roads here?', 'What is visible?'), use single_image_vqa.\n"
            "7. If the intent is unclear, vague, or cannot be safely mapped to the above (e.g., 'Analyze this', 'Look at this'), set ambiguous=true and primary_task=unknown.\n\n"
            "CRITICAL CONSTRAINTS:\n"
            "- DO NOT invent modality (if not explicitly mentioned, use unspecified).\n"
            "- DO NOT assume temporal intent unless they mention time, changes, or comparisons across dates.\n"
            "- DO NOT select implementation models or tools. Just classify the user's intent.\n\n"
            f"Query: \"{query}\"\n\n"
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
        
        # If there's leading/trailing garbage, try to find the outermost braces
        start = text.find("{")
        end = text.rfind("}")
        
        if start != -1 and end != -1 and end >= start:
            text = text[start:end+1]
            
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def extract(self, query: str) -> TaskSpec:
        """
        Runs the extraction prompt and validates the resulting TaskSpec.
        """
        # Default fallback for extreme failures
        fallback_spec = TaskSpec(
            query=query,
            primary_task="unknown",
            ambiguous=True
        )
        
        prompt = self._build_prompt(query)
        
        # We use strict generation to reduce JSON hallucinations
        result = self.inference_engine.generate(prompt, max_new_tokens=512, temperature=0.01)
        
        if result["status"] != "ok" or not result["response"]:
            return fallback_spec
            
        parsed_dict = self._parse_json(result["response"])
        
        if not parsed_dict:
            return fallback_spec
            
        try:
            # Pydantic validation
            spec = TaskSpec(**parsed_dict)
            return spec
        except Exception:
            # If Pydantic fails (e.g. invalid enums), we fallback safely
            return fallback_spec
