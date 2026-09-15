import asyncio
from job_manager import MockQwenEngine
from agent.selector import AgentToolSelector
from mc4c.query_schema import QueryRepresentation
from job_manager import setup_default_registry
import job_manager

registry = setup_default_registry()
selector = AgentToolSelector(registry, MockQwenEngine())

qi = job_manager.QueryIntelligenceResult(
    original_query="change detection smoke_test",
    primary_task_spec=job_manager.TaskSpec(
        query="change detection smoke_test",
        primary_task="change_detection",
        target_entities=[],
        required_modalities=["optical"],
        spatial_output_required=True,
        textual_output_required=True,
        temporal_requirement="before_after"
    ),
    is_compound=False,
    subtasks=[],
    observation_requirements=[job_manager.ObservationRequirement(requirement_id="req_1", source_subtasks=[], minimum_observations=2, maximum_observations=2, required_modalities=["optical"], temporal_relationship="before_after")],
    ambiguous=False
)

selector.select_tool(qi)

