from typing import List, Dict, Any, Tuple
from ..schemas import BenchmarkSample
from ..metrics import Metric

class BaseBenchmarkAdapter:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_samples(self, limit: int = 10) -> List[BenchmarkSample]:
        raise NotImplementedError

    def build_satquery_request(self, sample: BenchmarkSample) -> Tuple[str, List[Tuple[str, bytes]]]:
        """
        Returns (query_string, [(filename, file_bytes)])
        """
        raise NotImplementedError

    def parse_prediction(self, job_result: Dict[str, Any], status: str) -> Any:
        if status != "DONE":
            return None
        return job_result.get("final_answer")

    def get_metrics(self) -> List[Metric]:
        raise NotImplementedError
