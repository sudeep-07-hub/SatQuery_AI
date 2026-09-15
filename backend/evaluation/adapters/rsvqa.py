import json
import os
from typing import List, Dict, Any, Tuple
from .base import BaseBenchmarkAdapter
from ..schemas import BenchmarkSample
from ..metrics import ExactMatchMetric

class RSVQAAdapter(BaseBenchmarkAdapter):
    def __init__(self, data_path: str):
        super().__init__(data_path)
        self.metrics = [ExactMatchMetric()]

    def load_samples(self, limit: int = 10) -> List[BenchmarkSample]:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"RSVQA data file not found: {self.data_path}")
            
        with open(self.data_path, "r") as f:
            data = json.load(f)
            
        samples = []
        for i, item in enumerate(data):
            if i >= limit:
                break
            samples.append(BenchmarkSample(
                benchmark_id="RSVQA",
                sample_id=item.get("id", str(i)),
                task_type="vqa",
                query=item.get("question", ""),
                observations=[{"path": item.get("img_path", "image1.tif")}],
                ground_truth=item.get("answer", ""),
                expected_output_type="text"
            ))
        return samples

    def build_satquery_request(self, sample: BenchmarkSample) -> Tuple[str, List[Tuple[str, bytes]]]:
        query = f"rsvqa_eval: {sample.query}"
        
        # In a real run, we would load the actual image bytes.
        # For fixture runs or if the real image is unavailable, we use dummy bytes
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
        return self.metrics
