from typing import List, Dict, Optional
from .schemas import ToolSpec

class AgentToolRegistry:
    """
    Authoritative Registry of Agent Capabilities.
    Provides deterministic registration, lookup, and filtering.
    Does NOT execute tools or instantiate models.
    """
    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}
        
    def register(self, tool: ToolSpec) -> None:
        """Register a new capability. Rejects duplicates deterministically."""
        if tool.tool_id in self._tools:
            raise ValueError(f"Duplicate tool_id registered: {tool.tool_id}")
        self._tools[tool.tool_id] = tool
        
    def get(self, tool_id: str) -> Optional[ToolSpec]:
        """Retrieve a specific capability by ID."""
        return self._tools.get(tool_id)
        
    def list_capabilities(self) -> List[ToolSpec]:
        """Returns all capabilities in deterministic (sorted by ID) order."""
        # Sort by key to ensure deterministic ordering
        return [self._tools[k] for k in sorted(self._tools.keys())]
        
    def find_by_task(self, task_type: str) -> List[ToolSpec]:
        """
        Returns all capabilities that explicitly support the given Phase 2 task_type.
        """
        candidates = []
        for tool in self.list_capabilities():
            if task_type in tool.supported_tasks:
                candidates.append(tool)
        return candidates

    def validate_call(self, call: 'ToolCall') -> None:
        """
        Validates a ToolCall against the registry.
        Ensures the tool_id is known and the capability is enabled.
        Raises ValueError if validation fails.
        """
        tool = self.get(call.tool_id)
        if not tool:
            raise ValueError(f"Validation failed: Unknown tool_id '{call.tool_id}'")
        
        if not tool.enabled:
            raise ValueError(f"Validation failed: Capability '{call.tool_id}' is disabled or unimplemented (status: {tool.status})")
