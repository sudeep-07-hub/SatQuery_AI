import pytest
from mc5_evidence.evidence_graph import EvidenceGraph

def test_evidence_graph_nodes_and_edges():
    graph = EvidenceGraph()
    evidence_obj = {
        "evidence_id": "test_job_mc4b_0",
        "claim": "Change detected.",
        "evidence_type": "bitemporal_change_detection",
        "spatial_region": {"type": "Polygon", "coordinates": [[[0,0], [0,1], [1,1], [1,0], [0,0]]]},
        "modality": "sar",
        "modality_contribution": {"sar": 1.0},
        "timestamp": {"t1": "t1", "t2": "t2"},
        "source_model": "CHANGE_MAMBA_TOOL",
        "source_input": {"image_1": "1.tif", "image_2": "2.tif"},
        "confidence": 0.9,
        "processing_parameters": {}
    }
    
    eid = graph.insert_evidence(evidence_obj, "query_1")
    
    data = graph.to_dict()
    nodes = data["nodes"]
    edges = data["edges"]
    
    # We should have nodes for: evidence, claim, model, modality, timestamp, region, query
    node_types = {n["type"] for n in nodes}
    assert "evidence" in node_types
    assert "claim" in node_types
    assert "model" in node_types
    assert "modality" in node_types
    assert "timestamp" in node_types
    assert "region" in node_types
    assert "query" in node_types
    
    # Verify honest edges
    edge_rels = {e["relationship"] for e in edges}
    assert "supports" in edge_rels
    assert "derived_from" in edge_rels
    assert "overlaps" in edge_rels
    
    # Omitted edges should not exist
    assert "contradicts" not in edge_rels
    assert "corroborates" not in edge_rels
    assert "precedes" not in edge_rels
