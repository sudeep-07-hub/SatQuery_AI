import pytest
import os
import asyncio
from backend.evaluation.runner import run_evaluation

def test_rsvqa_fixture_evaluation():
    fixture_path = os.path.join(os.path.dirname(__file__), "..", "evaluation", "fixtures", "rsvqa_fixture.json")
    result = asyncio.run(run_evaluation("rsvqa", fixture_path, execution_mode="fixture"))
    
    assert result.benchmark_id == "rsvqa"
    assert len(result.predictions) == 2
    for pred in result.predictions:
        assert pred.execution_mode == "fixture"
        # The mock synthesis for RSVQA returns 'yes' for the first item
    
    assert "exact_match" in result.metrics

def test_vrsbench_fixture_evaluation():
    fixture_path = os.path.join(os.path.dirname(__file__), "..", "evaluation", "fixtures", "vrsbench_fixture.json")
    result = asyncio.run(run_evaluation("vrsbench", fixture_path, execution_mode="fixture"))
    
    assert result.benchmark_id == "vrsbench"
    assert len(result.predictions) == 3
    
    # Grounding metric should return -1.0 indicating unsupported
    assert result.metrics.get("iou_semantic_grounding", 0) == -1.0

def test_cdvqa_fixture_evaluation():
    fixture_path = os.path.join(os.path.dirname(__file__), "..", "evaluation", "fixtures", "cdvqa_fixture.json")
    result = asyncio.run(run_evaluation("cdvqa", fixture_path, execution_mode="fixture"))
    
    assert result.benchmark_id == "cdvqa"
    assert len(result.predictions) == 1
    
    assert "exact_match" in result.metrics
