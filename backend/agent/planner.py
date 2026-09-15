import uuid
from typing import List, Dict, Set
from collections import defaultdict

from qwen.pipeline import QueryIntelligenceResult
from agent.schemas import ToolCall, WorkflowPlan, WorkflowDependency
from agent.registry import AgentToolRegistry

class WorkflowPlanner:
    """
    Deterministic DAG construction and validation layer.
    Transforms independent bound ToolCalls into an executable Stage-based WorkflowPlan.
    """
    
    def build_workflow(
        self, 
        calls: List[ToolCall], 
        qi_result: QueryIntelligenceResult, 
        registry: AgentToolRegistry
    ) -> WorkflowPlan:
        
        errors = []
        call_map: Dict[str, ToolCall] = {}
        subtask_map: Dict[str, str] = {} # maps subtask_id to call_id
        
        # 1. Node Validation & Map Construction
        for call in calls:
            # Prevent duplicate calls in our map directly
            if call.call_id in call_map:
                errors.append(f"Duplicate ToolCall ID encountered: {call.call_id}")
                continue
                
            try:
                registry.validate_call(call)
            except ValueError as e:
                errors.append(f"ToolCall {call.call_id} is invalid: {str(e)}")
                
            tool_spec = registry.get(call.tool_id)
            if not tool_spec or not tool_spec.enabled:
                errors.append(f"ToolCall {call.call_id} references disabled or missing capability: {call.tool_id}")
                
            call_map[call.call_id] = call
            for st_id in call.source_subtask_ids:
                subtask_map[st_id] = call.call_id

        # 2. Extract Phase 2 Subtask Metadata
        st_depends_on = {}
        for st in qi_result.subtasks:
            st_depends_on[st.subtask_id] = st.depends_on
            
        # 3. Edge Construction
        dependencies: List[WorkflowDependency] = []
        dep_set = set() # To detect duplicate edges
        
        for call in calls:
            for st_id in call.source_subtask_ids:
                depends_on_list = st_depends_on.get(st_id, [])
                for upstream_st_id in depends_on_list:
                    # Resolve which call handles the upstream subtask
                    upstream_call_id = subtask_map.get(upstream_st_id)
                    if not upstream_call_id:
                        errors.append(f"Dangling reference: ToolCall {call.call_id} depends on subtask {upstream_st_id}, which has no corresponding ToolCall.")
                        continue
                        
                    edge = (upstream_call_id, call.call_id)
                    if edge not in dep_set:
                        dep_set.add(edge)
                        dependencies.append(WorkflowDependency(
                            source_call_id=upstream_call_id,
                            target_call_id=call.call_id,
                            dependency_type="SEMANTIC_DEPENDENCY"
                        ))
                        
        # Support for Data Dependencies explicitly parsed from tool bindings, if any
        # (Though our schema is primarily observation bindings, if a tool result was referenced, it would be checked here).
        for call in calls:
            for role, binding in call.input_bindings.items():
                if binding.requirement_id and binding.requirement_id.startswith("tool_result:"):
                    # This is just a conceptual example based on the prompt's data_flow suggestion.
                    upstream_call_id = binding.requirement_id.split(":")[1]
                    if upstream_call_id not in call_map:
                        errors.append(f"Dangling reference: ToolCall {call.call_id} has a data dependency on unknown call {upstream_call_id}")
                        continue
                        
                    edge = (upstream_call_id, call.call_id)
                    if edge not in dep_set:
                        dep_set.add(edge)
                        dependencies.append(WorkflowDependency(
                            source_call_id=upstream_call_id,
                            target_call_id=call.call_id,
                            dependency_type="DATA_DEPENDENCY",
                            target_input=role
                        ))

        # 4. Graph Construction for DAG Validation
        adj = defaultdict(list)
        in_degree = {cid: 0 for cid in call_map}
        
        for dep in dependencies:
            # Dangling reference check explicitly for manually injected dependencies
            if dep.source_call_id not in call_map:
                errors.append(f"Dangling source reference in dependency: {dep.source_call_id}")
                continue
            if dep.target_call_id not in call_map:
                errors.append(f"Dangling destination reference in dependency: {dep.target_call_id}")
                continue
                
            if dep.source_call_id == dep.target_call_id:
                errors.append(f"Self-dependency detected for {dep.source_call_id}")
                continue
                
            adj[dep.source_call_id].append(dep.target_call_id)
            in_degree[dep.target_call_id] += 1
            
            # ToolSpec Output Validation for Data Dependencies
            if dep.dependency_type == "DATA_DEPENDENCY" and dep.source_output:
                tool_spec = registry.get(call_map[dep.source_call_id].tool_id)
                if tool_spec and dep.source_output not in tool_spec.output_types:
                    errors.append(f"Dependency output '{dep.source_output}' not declared by upstream ToolSpec {tool_spec.tool_id}")

        if errors:
            return WorkflowPlan(
                workflow_id=str(uuid.uuid4()),
                status="invalid",
                calls=call_map,
                dependencies=dependencies,
                execution_stages=[],
                errors=errors
            )

        # 5. Cycle Detection and Topological Staging
        # Kahn's Algorithm
        stages = []
        zero_in_degree = [cid for cid, deg in in_degree.items() if deg == 0]
        processed_count = 0
        
        while zero_in_degree:
            # The current zero_in_degree nodes form a stage, as they are independent of each other
            # Sort for deterministic stage generation
            current_stage = sorted(zero_in_degree)
            stages.append(current_stage)
            processed_count += len(current_stage)
            
            next_zero_in_degree = []
            for cid in current_stage:
                for target in adj[cid]:
                    in_degree[target] -= 1
                    if in_degree[target] == 0:
                        next_zero_in_degree.append(target)
            zero_in_degree = next_zero_in_degree
            
        if processed_count != len(call_map):
            errors.append("Cycle detected in dependency graph.")
            return WorkflowPlan(
                workflow_id=str(uuid.uuid4()),
                status="invalid",
                calls=call_map,
                dependencies=dependencies,
                execution_stages=[],
                errors=errors
            )
            
        return WorkflowPlan(
            workflow_id=str(uuid.uuid4()),
            status="ready",
            calls=call_map,
            dependencies=dependencies,
            execution_stages=stages,
            errors=[]
        )
