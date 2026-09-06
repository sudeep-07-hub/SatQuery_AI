"""
tool_registry.py — MC3.1 Tool Registry.

Maintains a validated registry of all available specialist tools.
Each tool entry must conform to the MC3.1 schema.
"""

from typing import Dict, List, Optional

# Schema validation keys
REQUIRED_TOOL_FIELDS = {
    "name", "task_capabilities", "supported_modalities",
    "input_constraints", "spatial_resolution_range",
    "temporal_requirements", "output_types",
    "expected_confidence_characteristics", "computational_cost",
    "preconditions", "postconditions",
}


class ToolRegistry:
    """Registry of MC3.1 Tool entries."""

    def __init__(self):
        self._tools: Dict[str, Dict] = {}

    def register(self, tool_entry: Dict) -> None:
        """Register a tool, validating against MC3.1 schema."""
        missing = REQUIRED_TOOL_FIELDS - set(tool_entry.keys())
        if missing:
            raise ValueError(
                f"Tool entry missing required fields: {missing}"
            )
        name = tool_entry["name"]
        self._tools[name] = tool_entry

    def get(self, name: str) -> Optional[Dict]:
        return self._tools.get(name)

    def find_by_capability(self, capability: str) -> List[Dict]:
        """Find all tools that declare a given capability."""
        return [
            t for t in self._tools.values()
            if capability in t.get("task_capabilities", [])
        ]

    def find_by_temporal_requirement(self, requirement: str) -> List[Dict]:
        """Find all tools matching a temporal requirement."""
        return [
            t for t in self._tools.values()
            if requirement in t.get("temporal_requirements", [])
        ]

    def all_tools(self) -> List[Dict]:
        return list(self._tools.values())
