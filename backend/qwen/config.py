import os

class QwenConfig:
    """
    Centralized configuration for the Qwen3 4B Inference Foundation.
    """
    MODEL_ID = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507")
    
    # Generation defaults
    DEFAULT_MAX_NEW_TOKENS = int(os.getenv("QWEN_MAX_NEW_TOKENS", 256))
    DEFAULT_TEMPERATURE = float(os.getenv("QWEN_TEMPERATURE", 0.1))
    
    # We prefer strict deterministic generation for infrastructure/smoke tests.
    DEFAULT_DO_SAMPLE = False
    
    # Memory/Runtime
    # If device is "auto", HuggingFace accelerate will automatically balance memory
    DEFAULT_DEVICE = os.getenv("QWEN_DEVICE", "auto")
