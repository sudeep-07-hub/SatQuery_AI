import json
import os
from typing import List, Dict, Any, Tuple
from .base import BaseBenchmarkAdapter
from ..schemas import BenchmarkSample
from ..metrics import ExactMatchMetric, FallbackTextMetric, SemanticGroundingMetric

class VRSBenchAdapter(BaseBenchmarkAdapter):
    def __init__(self, data_path: str):
        super().__init__(data_path)
        self.vqa_metrics = [ExactMatchMetric()]
        self.captioning_metrics = [FallbackTextMetric()]
        self.grounding_metrics = [SemanticGroundingMetric()]

    def load_samples(self, limit: int = 10) -> List[BenchmarkSample]:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"VRSBench data file not found: {self.data_path}")
            
        with open(self.data_path, "r") as f:
            data = json.load(f)
            
        samples = []
        for i, item in enumerate(data):
            if i >= limit:
                break
                
            task_type = item.get("task_type", "vqa")
            samples.append(BenchmarkSample(
                benchmark_id="VRSBench",
                sample_id=item.get("id", str(i)),
                task_type=task_type,
                query=item.get("question", item.get("prompt", "")),
                observations=[{"path": item.get("img_path", "image1.tif")}],
                ground_truth=item.get("answer", item.get("reference", "")),
                expected_output_type="spatial" if task_type == "grounding" else "text"
            ))
        return samples

    def build_satquery_request(self, sample: BenchmarkSample) -> Tuple[str, List[Tuple[str, bytes]]]:
        query = f"vrsbench_eval: {sample.query}"
        
        files = []
        for obs in sample.observations:
            img_path = obs.get("path", "image1.tif")
            if os.path.exists(img_path):
                with open(img_path, "rb") as img_f:
                    files.append((os.path.basename(img_path), img_f.read()))
            else:
                files.append((os.path.basename(img_path), b"dummy"))
                
        return query, files

    def get_metrics(self) -> List[Any]:
        # Return a union or decide dynamically in the runner
        return self.vqa_metrics + self.captioning_metrics + self.grounding_metrics
