from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from qwen.schemas import ObservationRequirement

class ToolSpec(BaseModel):
    """
    Formal capability contract describing an available specialist tool.
    This defines WHAT the tool can do, mapped to Phase 2 semantic tasks,
    without executing or describing HOW it does it.
    """
    tool_id: str = Field(description="Unique, stable identifier for the capability")
    name: str = Field(description="Human-readable name")
    description: str = Field(description="High-level description of capability")
    
    # Task mapping (e.g. ['single_image_vqa', 'captioning'])
    supported_tasks: List[str] = Field(description="Phase 2 semantic tasks this capability supports")
    
    # Pre-requisite Observation constraints (must match Phase 2 ObservationRequirement to be viable)
    required_observations: ObservationRequirement = Field(description="Explicit observation prerequisites")
    
    # Output description (e.g., ['textual_answer', 'spatial_evidence'])
    output_types: List[str] = Field(description="Semantic output components provided by this capability")
    
    # State flags
    status: Literal["implemented", "partial", "mocked", "disconnected", "unavailable"] = Field(
        description="The actual codebase implementation status"
    )
    enabled: bool = Field(default=False, description="Whether this capability is exposed for agent execution")

class InputBinding(BaseModel):
    """
    Semantic contract for future input binding.
    Expresses WHAT input is expected without actually binding to physical files.
    """
    requirement_id: Optional[str] = Field(default=None, description="References a Phase 2 ObservationRequirement ID")
    semantic_role: Optional[str] = Field(default=None, description="Role of the input (e.g. 'primary_observation')")
    observation_id: Optional[str] = Field(default=None, description="References a specific MC1 Observation ID")

BindingStatus = Literal["SUFFICIENT", "INSUFFICIENT", "INCOMPATIBLE", "AMBIGUOUS"]

class BindingResult(BaseModel):
    """
    Result of deterministic observation sufficiency evaluation.
    """
    status: BindingStatus
    requirement_id: str
    bound_call: Optional['ToolCall'] = None
    failed_constraints: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    unresolved_roles: List[str] = Field(default_factory=list)
    candidate_observation_ids: List[str] = Field(default_factory=list)

from pydantic import model_validator
from typing import Any, Dict

class ToolCall(BaseModel):
    """
    Formal representation of a requested capability invocation.
    Defines WHAT capability is being requested, WITH WHAT validated arguments, 
    and AGAINST WHICH semantic inputs.
    """
    call_id: str = Field(description="Unique identifier for this call")
    tool_id: str = Field(description="References a registered ToolSpec capability ID")
    
    # Provenance tracking
    source_subtask_ids: List[str] = Field(default_factory=list, description="IDs of the Phase 2 subtasks triggering this call")
    
    # Arguments
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Structured arguments for the tool")
    
    # Input bindings
    input_bindings: Dict[str, InputBinding] = Field(default_factory=dict, description="Semantic input references")
    
    # State tracking
    state: Literal["pending", "validated", "executing", "succeeded", "failed", "skipped"] = Field(
        default="pending", description="Execution state of the call"
    )

    @model_validator(mode='after')
    def validate_security(self) -> 'ToolCall':
        """
        Ensure arguments do not contain executable payloads, shell commands, or arbitrary paths.
        """
        # Extremely basic heuristic blocklist to prevent flagrant code execution injections.
        blocked_substrings = [
            "os.system", "subprocess", "eval(", "exec(", "import ", "sys.modules", 
            "rm -rf", "wget ", "curl ", "/bin/sh", "/bin/bash", "open("
        ]
        
        def _check_val(val: Any):
            if isinstance(val, str):
                v_lower = val.lower()
                for b in blocked_substrings:
                    if b in v_lower:
                        raise ValueError(f"Security violation: executable or blocked string pattern detected in arguments: '{b}'")
            elif isinstance(val, dict):
                for v in val.values():
                    _check_val(v)
            elif isinstance(val, list):
                for v in val:
                    _check_val(v)
                    
        for val in self.arguments.values():
            _check_val(val)
            
        return self

WorkflowDependencyType = Literal["DATA_DEPENDENCY", "SEMANTIC_DEPENDENCY"]

class WorkflowDependency(BaseModel):
    source_call_id: str = Field(description="The call_id that must execute first")
    target_call_id: str = Field(description="The call_id that depends on the source")
    dependency_type: WorkflowDependencyType
    source_output: Optional[str] = Field(default=None, description="The specific output from the source if this is a data dependency")
    target_input: Optional[str] = Field(default=None, description="The specific input role required by the target if this is a data dependency")

class WorkflowPlan(BaseModel):
    workflow_id: str = Field(description="Unique application-generated identifier for this workflow")
    status: Literal["draft", "validated", "ready", "invalid"] = Field(description="The validation/execution state of the workflow")
    calls: Dict[str, ToolCall] = Field(description="A dictionary mapping call_ids to validated ToolCalls")
    dependencies: List[WorkflowDependency] = Field(default_factory=list, description="Explicit DAG edges representing execution requirements")
    execution_stages: List[List[str]] = Field(default_factory=list, description="Ordered list of stages; each stage contains independent call_ids")
    errors: List[str] = Field(default_factory=list, description="Structural or validation failures encountered during planning")

ToolResultStatus = Literal["succeeded", "failed", "skipped"]

class ToolResult(BaseModel):
    call_id: str = Field(description="The ID of the executed ToolCall")
    tool_id: str = Field(description="The capability executed")
    status: ToolResultStatus = Field(description="Execution result status")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Structured output metadata (no raw tensors)")
    evidence_references: List[str] = Field(default_factory=list, description="List of generated Evidence Object IDs")
    error_information: Optional[str] = Field(default=None, description="Exception or failure reason if failed/skipped")
    execution_metadata: Dict[str, Any] = Field(default_factory=dict, description="Timing and execution context")

FailureType = Literal["INPUT_INVALID", "INPUT_INSUFFICIENT", "TOOL_UNAVAILABLE", "TOOL_DISABLED", "TOOL_EXECUTION_FAILED", "TOOL_OUTPUT_INVALID", "DEPENDENCY_FAILED", "WORKFLOW_INVALID", "RESOURCE_LIMIT", "UNKNOWN_FAILURE"]

class FailureContext(BaseModel):
    workflow_id: str
    failed_call_id: str
    failed_tool_id: str
    failure_type: FailureType
    failure_reason: str
    available_tools: List[str]
    remaining_replans: int

class ReplanRequest(BaseModel):
    recovery_possible: bool
    reason: str
    replacement_tool_id: Optional[str] = None
    arguments: Dict[str, Any] = Field(default_factory=dict)

AgentStateStatus = Literal["pending", "running", "succeeded", "failed", "completed"]

class AgentState(BaseModel):
    workflow_id: str = Field(description="The ID of the currently executing WorkflowPlan")
    current_stage_index: int = Field(default=0, description="The index of the currently executing stage")
    completed_calls: List[str] = Field(default_factory=list)
    pending_calls: List[str] = Field(default_factory=list)
    failed_calls: List[str] = Field(default_factory=list)
    skipped_calls: List[str] = Field(default_factory=list)
    results: Dict[str, ToolResult] = Field(default_factory=dict, description="Map of call_id to ToolResult")
    execution_trace: List[Dict[str, Any]] = Field(default_factory=list, description="Auditable trace of execution events")
    status: AgentStateStatus = Field(default="pending", description="Overall workflow execution status")
    
    # Budget tracking
    replan_count: int = Field(default=0, description="Number of replans executed")
    max_replans: int = Field(default=3, description="Maximum allowed replans")
    tool_call_count: int = Field(default=0, description="Total tools executed")
    max_tool_calls: int = Field(default=10, description="Maximum allowed tool executions")
    replan_history: List[Dict[str, Any]] = Field(default_factory=list, description="History of replan cycles")

# --- Phase 7.3 Final Answer Schemas ---

class FinalAnswerRequest(BaseModel):
    user_query: str
    verified_evidence: List[Dict[str, Any]]
    rejected_claims: List[Dict[str, Any]]
    observation_summary: List[Dict[str, Any]]
    execution_status: str
    uncertainty: str

class FinalAnswerResult(BaseModel):
    status: str
    answer: str
    evidence_ids: List[str]
    execution_metadata: Dict[str, Any]
    model_id: str
