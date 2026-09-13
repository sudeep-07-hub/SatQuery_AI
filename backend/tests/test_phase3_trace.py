import pytest
import os
import tempfile
import json
from mc8_export.exporter import export_json_trace

def test_json_trace_export_structured_trace():
    with tempfile.TemporaryDirectory() as tmpdir:
        job = {
            "job_id": "test_job_123",
            "status": "DONE",
            "progress_trace": [{"stage": "DONE", "message": "done"}],
            "evidence_objects": [{"evidence_id": "ev1", "claim": "claim 1"}],
            "evidence_graph": {"nodes": [], "edges": []}
        }
        
        filepath = export_json_trace(job, tmpdir)
        assert os.path.exists(filepath)
        
        with open(filepath, "r") as f:
            data = json.load(f)
            
        assert data["job_id"] == "test_job_123"
        assert "structured_trace" in data
        assert len(data["structured_trace"]["evidence_objects"]) == 1
        assert data["structured_trace"]["evidence_objects"][0]["evidence_id"] == "ev1"
        assert "evidence_graph" in data["structured_trace"]
