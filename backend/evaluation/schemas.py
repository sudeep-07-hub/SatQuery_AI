from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class BenchmarkSample(BaseModel):
    benchmark_id: str
    sample_id: str
    task_type: str
    query: str
    observations: List[Dict[str, Any]]
    ground_truth: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)
    expected_output_type: str

class BenchmarkPrediction(BaseModel):
    benchmark_id: str
    sample_id: str
    task_type: str
    prediction: Any
    evidence_ids: List[str] = Field(default_factory=list)
    execution_status: str
    execution_mode: str
    model_backend: str
    model_id: str
    latency_ms: int
    error_code: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class EvaluationResult(BaseModel):
    benchmark_id: str
    metrics: Dict[str, float]
    predictions: List[BenchmarkPrediction]
    summary: str
    provenance: Dict[str, Any]
