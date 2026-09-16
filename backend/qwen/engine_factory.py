"""
engine_factory.py — Selects the Qwen3 runtime for the application pipeline.

Backends (env SATQUERY_QWEN_BACKEND, default "ollama"):
  ollama        Qwen3-4B Q4_K_M GGUF served by a local Ollama daemon (~2.5 GB, fits a 16 GB Mac)
  transformers  Full bf16 Qwen/Qwen3-4B-Instruct-2507 via Hugging Face (~8 GB RAM)
  none          No LLM; the deterministic tool-registry planner is used
"""
import os
from typing import Optional, Tuple

DEFAULT_BACKEND = "ollama"


def resolve_backend(requested: Optional[str] = None) -> str:
    backend = (requested or os.getenv("SATQUERY_QWEN_BACKEND") or DEFAULT_BACKEND).strip().lower()
    if backend not in ("ollama", "transformers", "none"):
        raise ValueError(f"Unknown Qwen backend '{backend}' (expected ollama, transformers or none)")
    return backend


def model_label(backend: str) -> str:
    if backend == "ollama":
        return f"{os.getenv('SATQUERY_OLLAMA_MODEL', 'qwen3:4b')} (ollama, Q4_K_M)"
    if backend == "transformers":
        return "Qwen3-4B-Instruct-2507 (transformers)"
    return "none (registry rules)"


def create_engine(backend: str):
    """Instantiate (but do not load) the engine for a backend. Returns None for 'none'."""
    if backend == "ollama":
        from qwen.ollama_inference import OllamaQwen3Inference
        return OllamaQwen3Inference(
            model_name=os.getenv("SATQUERY_OLLAMA_MODEL", "qwen3:4b"),
            endpoint=os.getenv("SATQUERY_OLLAMA_URL", "http://localhost:11434"),
        )
    if backend == "transformers":
        from qwen.inference import Qwen3Inference
        return Qwen3Inference()
    return None


def load_engine(backend: str) -> Tuple[Optional[object], Optional[str]]:
    """Create and load an engine. Returns (engine, error). engine is None when unavailable."""
    engine = create_engine(backend)
    if engine is None:
        return None, "LLM disabled (SATQUERY_QWEN_BACKEND=none)"
    try:
        engine.load()
        return engine, None
    except Exception as e:
        return None, str(e)
