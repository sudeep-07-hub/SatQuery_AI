import uuid
from typing import Dict, Any, Optional

from agent.schemas import (
    WorkflowPlan, AgentState, FailureContext, ToolCall, 
    InputBinding, WorkflowDependency
)
from agent.controller import AgentController
from agent.recovery_planner import QwenRecoveryPlanner
from agent.registry import AgentToolRegistry
from agent.planner import WorkflowPlanner
from qwen.pipeline import QueryIntelligenceResult

class RecoveryManager:
    """
    Wraps the AgentController with a bounded replanning loop.
    """
    def __init__(
        self, 
        controller: AgentController, 
        recovery_planner: QwenRecoveryPlanner,
        registry: AgentToolRegistry,
        planner: WorkflowPlanner
    ):
        self.controller = controller
        self.recovery_planner = recovery_planner
        self.registry = registry
        self.planner = planner

    def _append_trace(self, state: AgentState, event_type: str, status: str, **kwargs):
        trace_event = {
            "workflow_id": state.workflow_id,
            "event_type": event_type,
            "status": status,
            "timestamp": __import__('datetime').datetime.utcnow().isoformat()
        }
        trace_event.update(kwargs)
        state.execution_trace.append(trace_event)

    def execute_with_recovery(
        self, 
        plan: WorkflowPlan, 
        qi_result: QueryIntelligenceResult,
        mc1_profile: Dict, 
        image_tensors: Dict,
        query: str,
        state: Optional[AgentState] = None
    ) -> AgentState:
        
        while True:
            # 1. Execute current plan
            state = self.controller.execute(plan, mc1_profile, image_tensors, state)
            
            # Increment tool call counts based on newly processed in this cycle
            # (Wait, a better way is to sum completed + failed, but let's just use the state len)
            state.tool_call_count = len(state.completed_calls) + len(state.failed_calls)
            
            if state.status == "completed":
                return state
                
            if state.status == "failed":
                # Check budgets
                if state.tool_call_count >= state.max_tool_calls:
                    self._append_trace(state, "budget_exhausted", "failed", reason="max_tool_calls exceeded")
                    return state
                    
                if state.replan_count >= state.max_replans:
                    self._append_trace(state, "budget_exhausted", "failed", reason="max_replans exceeded")
                    return state
                
                # Identify first failed call to recover
                failed_call_id = state.failed_calls[-1] if state.failed_calls else None
                if not failed_call_id:
                    self._append_trace(state, "abstention", "failed", reason="Unrecoverable workflow validation or structure error")
                    return state
                    
                failed_call = plan.calls[failed_call_id]
                failed_result = state.results.get(failed_call_id)
                failure_reason = failed_result.error_information if failed_result else "Unknown execution error"
                
                # Check if identical retry was already attempted (naive heuristic)
                if any(h.get("failed_tool_id") == failed_call.tool_id and h.get("failed_reason") == failure_reason for h in state.replan_history):
                    self._append_trace(state, "abstention", "failed", reason="Repeated identical failure detected")
                    return state
                    
                # 2. Analyze & Propose Recovery
                self._append_trace(state, "failure_detected", "failed", call_id=failed_call_id, tool_id=failed_call.tool_id)
                
                available_tools = [t for t in self.registry.list_capabilities() if t.enabled]
                
                context = FailureContext(
                    workflow_id=state.workflow_id,
                    failed_call_id=failed_call_id,
                    failed_tool_id=failed_call.tool_id,
                    failure_type="TOOL_EXECUTION_FAILED",
                    failure_reason=failure_reason,
                    available_tools=[t.tool_id for t in available_tools],
                    remaining_replans=state.max_replans - state.replan_count
                )
                
                try:
                    replan = self.recovery_planner.propose_recovery(context, available_tools, query)
                except Exception as e:
                    self._append_trace(state, "replan_rejected", "failed", reason=f"Qwen error: {str(e)}")
                    state.status = "failed"
                    return state
                    
                if not replan.recovery_possible or not replan.replacement_tool_id:
                    self._append_trace(state, "abstention", "failed", reason=replan.reason)
                    state.status = "failed"
                    return state
                    
                if replan.replacement_tool_id == failed_call.tool_id:
                    self._append_trace(state, "replan_rejected", "failed", reason="Same-tool retry is disabled")
                    state.status = "failed"
                    return state
                    
                # 3. Construct Replacement Plan
                self._append_trace(state, "replan_requested", "running", proposed_tool=replan.replacement_tool_id)
                
                new_call_id = f"call_{uuid.uuid4().hex[:8]}"
                replacement_call = ToolCall(
                    call_id=new_call_id,
                    tool_id=replan.replacement_tool_id,
                    arguments=replan.arguments,
                    source_subtask_ids=failed_call.source_subtask_ids,
                    input_bindings=failed_call.input_bindings,
                    state="pending"
                )
                
                # Copy existing valid calls (completed, or pending/skipped that weren't the failure)
                # We'll actually just provide the active list of calls.
                new_calls = []
                for cid, call in plan.calls.items():
                    if cid == failed_call_id:
                        continue
                    if cid in state.skipped_calls:
                        # Skipped calls might need to be retried if their dependency is now fixed
                        # So we include them, and the controller will see them as pending.
                        # But wait, skipped_calls must be removed from state.skipped_calls to run again!
                        state.skipped_calls.remove(cid)
                    new_calls.append(call)
                new_calls.append(replacement_call)
                
                # Build new DAG
                new_plan = self.planner.build_workflow(new_calls, qi_result, self.registry)
                
                if new_plan.status != "ready":
                    self._append_trace(state, "replan_rejected", "failed", reason=f"Invalid new DAG: {new_plan.errors}")
                    state.status = "failed"
                    return state
                    
                # 4. Commit Replan
                state.replan_count += 1
                state.replan_history.append({
                    "failed_call_id": failed_call_id,
                    "failed_tool_id": failed_call.tool_id,
                    "failed_reason": failure_reason,
                    "replacement_tool_id": replan.replacement_tool_id,
                    "new_call_id": new_call_id
                })
                state.failed_calls.remove(failed_call_id)
                self._append_trace(state, "replan_accepted", "succeeded", replacement_tool=replan.replacement_tool_id)
                
                # Inherit workflow identity / revision tracking
                new_plan.workflow_id = plan.workflow_id
                plan = new_plan
                
                # Loop continues
