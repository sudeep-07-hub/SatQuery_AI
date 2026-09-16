import json
import uuid
from typing import Optional, List

from qwen.inference import Qwen3Inference
from qwen.schemas import TaskSpec, SubtaskSpec
from qwen.pipeline import QueryIntelligenceResult
from agent.schemas import ToolCall, InputBinding
from agent.registry import AgentToolRegistry


class QwenToolSelector:
    """
    Connects Phase 2 QueryIntelligenceResult to Phase 3 AgentToolRegistry using Qwen3.
    Constructs deterministic ToolCalls without executing models or binding physical inputs.
    """
    
    def __init__(self, inference_engine: Qwen3Inference, registry: AgentToolRegistry):
        self.inference = inference_engine
        self.registry = registry

    def _build_prompt(self, task, query_context: str, enabled_tools: list) -> str:
        """Construct a constrained prompt to instruct Qwen on tool selection."""
        tools_str = "\n".join([f"- {t.tool_id}: {t.description} (Tasks: {', '.join(t.supported_tasks)})" for t in enabled_tools])
        
        prompt = f"""You are an authoritative capability router for SatQuery AI.
Your ONLY job is to select the most appropriate capability from the registered tools based on the requested analytical task.

Do NOT invent tools. You MUST choose exactly ONE tool_id from the following list:
{tools_str}

Requested Task Profile:
- Primary Intent: {task.primary_task}
- Original Query Context: {query_context}
- Required Modalities: {task.required_modalities}
- Temporal Requirement: {task.temporal_requirement}

Respond with ONLY valid JSON matching this schema:
{{
  "tool_id": "the_selected_tool_id",
  "arguments": {{
    "query": "the original or refined query string"
  }}
}}
No markdown formatting, no explanations, no chain of thought. Just JSON.
"""
        return prompt

    def select_tool(self, intent: QueryIntelligenceResult) -> ToolCall:
        """Single-subtask convenience wrapper around select_tools()."""
        return self.select_tools(intent)[0]

    def select_tools(self, intent: QueryIntelligenceResult) -> List[ToolCall]:
        """
        Takes the deterministic Phase 2 intent and selects valid ToolCalls for each subtask.
        Handles ambiguity, malformed JSON, and validation failures safely.
        """
        if intent.ambiguous or (intent.primary_task_spec.ambiguous and not intent.is_compound):
            raise ValueError("Query intent is ambiguous. Safe fallback: Tool selection rejected.")
            
        enabled_tools = [t for t in self.registry.list_capabilities() if t.enabled]
        if not enabled_tools:
            raise ValueError("No enabled capabilities available in the registry.")
            
        # Ensure we have at least one subtask
        subtasks_to_process = intent.subtasks
        if not subtasks_to_process and intent.primary_task_spec.primary_task != "unknown":
            # Atomic intent without an explicit decomposition: the primary TaskSpec is the only subtask
            spec = intent.primary_task_spec
            subtasks_to_process = [SubtaskSpec(
                subtask_id="primary",
                description=spec.query,
                primary_task=spec.primary_task,
                target_entities=spec.target_entities,
                required_modalities=spec.required_modalities,
                temporal_requirement=spec.temporal_requirement,
                spatial_output_required=spec.spatial_output_required,
                textual_output_required=spec.textual_output_required,
            )]
        if not subtasks_to_process:
            raise ValueError("No subtasks available in the query decomposition.")
            
        calls = []
        for subtask in subtasks_to_process:
            prompt = self._build_prompt(subtask, intent.original_query, enabled_tools)
            result = self.inference.generate(prompt, temperature=0.1, max_new_tokens=256)
            
            if result["status"] != "ok":
                raise RuntimeError(f"Qwen3 Inference failed: {result.get('error')}")
                
            raw_output = result["response"]
            
            if raw_output.startswith("```json"):
                raw_output = raw_output[7:]
            if raw_output.startswith("```"):
                raw_output = raw_output[3:]
            if raw_output.endswith("```"):
                raw_output = raw_output[:-3]
            raw_output = raw_output.strip()
            
            try:
                parsed = json.loads(raw_output)
            except json.JSONDecodeError:
                raise ValueError(f"Malformed JSON from Qwen: {raw_output}")
                
            tool_id = parsed.get("tool_id")
            if not tool_id:
                raise ValueError("Missing 'tool_id' in Qwen output")
                
            arguments = parsed.get("arguments", {})
            
            call_id = f"call_{uuid.uuid4().hex[:8]}"
            
            bindings = {}
            if intent.observation_requirements:
                # Try to find a requirement matching this subtask
                matching_req = next((r for r in intent.observation_requirements if subtask.subtask_id in r.source_subtasks), None)
                if matching_req:
                    bindings["primary"] = InputBinding(requirement_id=matching_req.requirement_id)
                else:
                    # Fallback to the first available requirement
                    bindings["primary"] = InputBinding(requirement_id=intent.observation_requirements[0].requirement_id)
                    
            call = ToolCall(
                call_id=call_id,
                tool_id=tool_id,
                arguments=arguments,
                source_subtask_ids=[subtask.subtask_id],
                input_bindings=bindings,
                state="pending"
            )
            
            # Validate
            try:
                self.registry.validate_call(call)
            except ValueError as e:
                raise ValueError(f"Tool selection validation failed: {str(e)}")
                
            tool_spec = self.registry.get(tool_id)
            if subtask.primary_task not in tool_spec.supported_tasks:
                raise ValueError(f"Tool '{tool_id}' does not support task '{subtask.primary_task}'")
                
            calls.append(call)
            
        return calls
