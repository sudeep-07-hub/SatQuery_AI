"""
evidence_graph.py — Spatial Evidence Graph (MC5.2).

Nodes (7 types): Query, Claims, Models, Modalities, Regions, Timestamps, Evidence.
Edges (3 honest types): derived_from, overlaps, supports.
Explicitly omitted per known gaps: contradicts, corroborates, precedes.
"""

from typing import Dict, List, Optional, Tuple
import json
try:
    from shapely.geometry import shape
except ImportError:
    shape = None

class EvidenceGraph:
    """
    In-memory spatial evidence graph.
    Nodes are dicts with {"id", "type", "data"}.
    Edges are dicts with {"source", "target", "relationship"}.
    """

    def __init__(self):
        self._nodes: Dict[str, Dict] = {}
        self._edges: List[Dict] = []

    def add_node(self, node_id: str, node_type: str, data: Dict) -> None:
        if node_id not in self._nodes:
            self._nodes[node_id] = {"id": node_id, "type": node_type, "data": data}

    def add_edge(self, source_id: str, target_id: str, relationship: str) -> None:
        edge = {"source": source_id, "target": target_id, "relationship": relationship}
        if edge not in self._edges:
            self._edges.append(edge)

    def get_node(self, node_id: str) -> Optional[Dict]:
        return self._nodes.get(node_id)

    def insert_evidence(self, evidence_obj: Dict, query_id: str) -> str:
        """
        Insert an MC5.1 Evidence Object into the graph.

        Creates nodes for all 7 types:
        - Query (handled externally but linked here if passed)
        - Claims
        - Models
        - Modalities
        - Regions
        - Timestamps
        - Evidence
        
        Creates honest edges:
        - Evidence -> supports -> Claims
        - Evidence -> derived_from -> Models
        - Evidence -> derived_from -> Modalities
        - Evidence -> derived_from -> Timestamps
        - Evidence -> overlaps -> Regions
        """
        eid = evidence_obj["evidence_id"]

        # 1. Evidence node
        self.add_node(eid, "evidence", evidence_obj)

        # 2. Claim node
        claim_id = f"claim_{eid}"
        self.add_node(claim_id, "claim", {"text": evidence_obj.get("claim", "")})
        self.add_edge(eid, claim_id, "supports")

        # 3. Model node
        model_name = evidence_obj.get("source_model", "unknown")
        model_id = f"model_{model_name}"
        self.add_node(model_id, "model", {"name": model_name})
        self.add_edge(eid, model_id, "derived_from")

        # 4. Modality node
        modality = evidence_obj.get("modality", "optical")
        modality_id = f"modality_{modality}"
        self.add_node(modality_id, "modality", {"name": modality})
        self.add_edge(eid, modality_id, "derived_from")
        
        # 5. Timestamp node
        timestamp = evidence_obj.get("timestamp")
        # Ensure timestamp is hashable for ID, convert to string
        ts_str = json.dumps(timestamp, sort_keys=True) if isinstance(timestamp, dict) else str(timestamp)
        # simplistic ID, maybe hash it to avoid huge strings
        import hashlib
        ts_id = f"ts_{hashlib.md5(ts_str.encode()).hexdigest()[:8]}"
        self.add_node(ts_id, "timestamp", {"value": timestamp})
        self.add_edge(eid, ts_id, "derived_from")

        # 6. Region node
        region = evidence_obj.get("spatial_region")
        if region:
            region_id = f"region_{eid}"
            self.add_node(region_id, "region", {"spatial_region": region}) # UI needs node.data.spatial_region
            self.add_edge(eid, region_id, "overlaps")
            
            # Compute geometric overlaps between all regions if shape is available
            if shape is not None and "coordinates" in region:
                try:
                    s1 = shape(region)
                    for n_id, n in self._nodes.items():
                        if n["type"] == "region" and n_id != region_id:
                            other_geom = n["data"].get("spatial_region")
                            if other_geom and "coordinates" in other_geom:
                                s2 = shape(other_geom)
                                if s1.intersects(s2):
                                    self.add_edge(region_id, n_id, "overlaps")
                                    self.add_edge(n_id, region_id, "overlaps")
                except Exception:
                    pass

        # 7. Link Claim to Query if Query exists
        if query_id:
            self.add_node(query_id, "query", {"id": query_id})
            self.add_edge(claim_id, query_id, "supports")  # Spec uses supports between claim and query or similar

        return eid

    def to_dict(self) -> Dict:
        """Serialize graph matching contract vocabulary."""
        return {
            "nodes": list(self._nodes.values()),
            "edges": self._edges
        }

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)
