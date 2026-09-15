import json
from qwen.inference import Qwen3Inference
from agent.schemas import FailureContext, ReplanRequest, ToolSpec
from typing import List

class QwenRecoveryPlanner:
    def __init__(self, inference_engine: Qwen3Inference):
        self.inference = inference_engine

    def propose_recovery(self, context: FailureContext, available_tools: List[ToolSpec], query: str) -> ReplanRequest:
        tools_str = "\n".join([f"- {t.tool_id}: {t.description}" for t in available_tools])
        prompt = f"""You are a recovery planner for SatQuery AI.
An execution failure occurred. You must propose a recovery action.

Original Query: {query}
Failed Tool ID: {context.failed_tool_id}
Failure Type: {context.failure_type}
Failure Reason: {context.failure_reason}

Available Capabilities:
{tools_str}

If recovery is impossible or no alternative capability exists, you must abstain.
If recovery is possible, select EXACTLY ONE tool_id from the Available Capabilities list above.
DO NOT invent tools.

Respond with ONLY valid JSON matching this schema:
{{
  "recovery_possible": true/false,
  "reason": "Brief explanation",
  "replacement_tool_id": "tool_id or null",
  "arguments": {{
    "query": "the original or refined query"
  }}
}}
"""
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
            
        is_possible = parsed.get("recovery_possible", False)
        tool_id = parsed.get("replacement_tool_id")
        
        if is_possible and tool_id:
             valid_ids = [t.tool_id for t in available_tools]
             if tool_id not in valid_ids:
                 raise ValueError(f"Qwen hallucinated a replacement tool ID: {tool_id}")
                 
        return ReplanRequest(
            recovery_possible=is_possible,
            reason=parsed.get("reason", "No reason provided"),
            replacement_tool_id=tool_id if is_possible else None,
            arguments=parsed.get("arguments", {})
        )
