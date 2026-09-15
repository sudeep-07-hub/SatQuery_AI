from typing import Dict, Any, List
from datetime import datetime

from agent.schemas import WorkflowPlan, AgentState, ToolResult
from agent.registry import AgentToolRegistry
from agent.adapters import ToolExecutionAdapter

class AgentController:
    """
    Executes a WorkflowPlan using registered ToolExecutionAdapters.
    """
    def __init__(self, registry: AgentToolRegistry, adapters: Dict[str, ToolExecutionAdapter]):
        self.registry = registry
        self.adapters = adapters

    def _append_trace(self, state: AgentState, event_type: str, status: str, call_id: str = None, tool_id: str = None, error: str = None):
        trace_event = {
            "workflow_id": state.workflow_id,
            "event_type": event_type,
            "status": status,
            "timestamp": datetime.utcnow().isoformat()
        }
        if call_id:
            trace_event["call_id"] = call_id
        if tool_id:
            trace_event["tool_id"] = tool_id
        if error:
            trace_event["error"] = error
            
        state.execution_trace.append(trace_event)

    def execute(self, plan: WorkflowPlan, mc1_profile: Dict, image_tensors: Dict, state: AgentState = None) -> AgentState:
        
        if state is None:
            state = AgentState(
                workflow_id=plan.workflow_id,
                status="pending",
                pending_calls=list(plan.calls.keys())
            )
        else:
            # Sync pending_calls with the current plan, avoiding completed/skipped/failed
            state.workflow_id = plan.workflow_id
            state.pending_calls = [c for c in plan.calls if c not in state.completed_calls and c not in state.skipped_calls and c not in state.failed_calls]
            state.status = "pending"
            
        if plan.status != "ready":
            state.status = "failed"
            self._append_trace(state, "workflow_start", "failed", error=f"Invalid WorkflowPlan status: {plan.status}")
            return state

        self._append_trace(state, "workflow_start", "running")
        state.status = "running"
        
        for stage_idx, stage_calls in enumerate(plan.execution_stages):
            state.current_stage_index = stage_idx
            
            for call_id in stage_calls:
                # Execution Idempotency check
                if call_id in state.completed_calls or call_id in state.failed_calls or call_id in state.skipped_calls:
                    continue
                
                call = plan.calls[call_id]
                
                # Check downstream blocking (if any upstream semantic dependency failed)
                upstream_failed = False
                upstream_deps = [dep for dep in plan.dependencies if dep.target_call_id == call_id]
                
                for dep in upstream_deps:
                    if dep.source_call_id in state.failed_calls or dep.source_call_id in state.skipped_calls:
                        upstream_failed = True
                        break
                        
                if upstream_failed:
                    state.pending_calls.remove(call_id)
                    state.skipped_calls.append(call_id)
                    
                    result = ToolResult(
                        call_id=call_id,
                        tool_id=call.tool_id,
                        status="skipped",
                        error_information="Prerequisite dependency failed or skipped"
                    )
                    state.results[call_id] = result
                    self._append_trace(state, "tool_execution", "skipped", call_id, call.tool_id, error="Prerequisite failed")
                    continue
                
                # Check if capability is registered and enabled
                tool_spec = self.registry.get(call.tool_id)
                if not tool_spec or not tool_spec.enabled:
                    state.pending_calls.remove(call_id)
                    state.failed_calls.append(call_id)
                    
                    result = ToolResult(
                        call_id=call_id,
                        tool_id=call.tool_id,
                        status="failed",
                        error_information="Tool capability not found or disabled"
                    )
                    state.results[call_id] = result
                    self._append_trace(state, "tool_execution", "failed", call_id, call.tool_id, error="Tool disabled")
                    continue
                
                adapter = self.adapters.get(call.tool_id)
                if not adapter:
                    state.pending_calls.remove(call_id)
                    state.failed_calls.append(call_id)
                    
                    result = ToolResult(
                        call_id=call_id,
                        tool_id=call.tool_id,
                        status="failed",
                        error_information="Execution adapter not found"
                    )
                    state.results[call_id] = result
                    self._append_trace(state, "tool_execution", "failed", call_id, call.tool_id, error="No adapter")
                    continue
                
                # Execute
                self._append_trace(state, "tool_execution", "running", call_id, call.tool_id)
                
                try:
                    result = adapter.execute(call, mc1_profile, image_tensors)
                except Exception as e:
                    # Catch unhandled adapter exceptions
                    result = ToolResult(
                        call_id=call_id,
                        tool_id=call.tool_id,
                        status="failed",
                        error_information=f"Adapter raised unhandled exception: {str(e)}"
                    )
                
                state.pending_calls.remove(call_id)
                state.results[call_id] = result
                
                if result.status == "succeeded":
                    state.completed_calls.append(call_id)
                    self._append_trace(state, "tool_execution", "succeeded", call_id, call.tool_id)
                else:
                    state.failed_calls.append(call_id)
                    self._append_trace(state, "tool_execution", "failed", call_id, call.tool_id, error=result.error_information)
                    
        # Check Workflow completion
        if state.failed_calls:
            state.status = "failed"
            self._append_trace(state, "workflow_completion", "failed", error=f"{len(state.failed_calls)} calls failed")
        elif state.pending_calls:
             # Should not happen unless there's a disconnect in stage evaluation
             state.status = "failed"
             self._append_trace(state, "workflow_completion", "failed", error="Workflow exited with pending calls")
        else:
             state.status = "completed"
             self._append_trace(state, "workflow_completion", "succeeded")
             
        return state
