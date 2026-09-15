from typing import Dict, Type
from .adapters.base import BaseBenchmarkAdapter
from .adapters.rsvqa import RSVQAAdapter
from .adapters.vrsbench import VRSBenchAdapter
from .adapters.cdvqa import CDVQAAdapter

ADAPTERS: Dict[str, Type[BaseBenchmarkAdapter]] = {
    "rsvqa": RSVQAAdapter,
    "vrsbench": VRSBenchAdapter,
    "cdvqa": CDVQAAdapter,
}

def get_adapter(benchmark_name: str, data_path: str) -> BaseBenchmarkAdapter:
    benchmark_name = benchmark_name.lower()
    if benchmark_name not in ADAPTERS:
        raise ValueError(f"Unsupported benchmark: {benchmark_name}. Supported: {list(ADAPTERS.keys())}")
    return ADAPTERS[benchmark_name](data_path)
