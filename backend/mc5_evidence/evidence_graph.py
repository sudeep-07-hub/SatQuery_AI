"""
evidence_graph.py — Spatial Evidence Graph (MC5).

Nodes: Query, Claims, Models, Modalities, Regions, Timestamps, Evidence.
Edges: supports, contradicts, corroborates, derived_from, overlaps, precedes.
"""

from typing import Dict, List, Optional, Tuple


class EvidenceGraph:
    """
    In-memory spatial evidence graph.
    Nodes are dicts with at least {"id", "type", "data"}.
    Edges are tuples (source_id, target_id, relationship).
    """

    def __init__(self):
        self._nodes: Dict[str, Dict] = {}
        self._edges: List[Tuple[str, str, str]] = []

    def add_node(self, node_id: str, node_type: str, data: Dict) -> None:
        self._nodes[node_id] = {"id": node_id, "type": node_type, "data": data}

    def add_edge(self, source_id: str, target_id: str, relationship: str) -> None:
        self._edges.append((source_id, target_id, relationship))

    def get_node(self, node_id: str) -> Optional[Dict]:
        return self._nodes.get(node_id)

    def get_edges_from(self, node_id: str) -> List[Tuple[str, str, str]]:
        return [(s, t, r) for s, t, r in self._edges if s == node_id]

    def get_edges_to(self, node_id: str) -> List[Tuple[str, str, str]]:
        return [(s, t, r) for s, t, r in self._edges if t == node_id]

    def find_contradictions(self) -> List[Tuple[str, str]]:
        """Find all pairs of nodes connected by 'contradicts' edges."""
        return [(s, t) for s, t, r in self._edges if r == "contradicts"]

    def insert_evidence(self, evidence_obj: Dict, query_id: str) -> str:
        """
        Insert an MC5.1 Evidence Object into the graph.

        Creates nodes for: evidence, claim, model, region.
        Creates edges: evidence→supports→claim, evidence→derived_from→model.
        """
        eid = evidence_obj["evidence_id"]

        # Evidence node
        self.add_node(eid, "evidence", evidence_obj)

        # Claim node
        claim_id = f"claim_{eid}"
        self.add_node(claim_id, "claim", {"text": evidence_obj.get("claim", "")})
        self.add_edge(eid, claim_id, "supports")

        # Model node
        model_name = evidence_obj.get("source_model", "unknown")
        model_id = f"model_{model_name}"
        if not self.get_node(model_id):
            self.add_node(model_id, "model", {"name": model_name})
        self.add_edge(eid, model_id, "derived_from")

        # Link to query
        self.add_edge(claim_id, query_id, "answers")

        # Spatial region node (if present)
        region = evidence_obj.get("spatial_region")
        if region:
            region_id = f"region_{eid}"
            self.add_node(region_id, "region", {"geometry": region})
            self.add_edge(eid, region_id, "overlaps")

        return eid

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)
