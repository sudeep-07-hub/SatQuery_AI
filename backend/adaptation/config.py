import os
from dataclasses import dataclass, field
import torch

@dataclass
class AdaptationConfig:
    # Projection settings
    visual_dim: int = 768
    text_dim: int = 2560
    embedding_dim: int = 512
    
    # Training settings
    batch_size: int = 1
    epochs: int = 5
    learning_rate: float = 1e-4
    temperature: float = 0.07
    
    # Paths
    features_dir: str = "/Users/sukesh/Desktop/satquery/backend/data/features"
    text_features_dir: str = "/Users/sukesh/Desktop/satquery/backend/data/text_features"
    checkpoints_dir: str = "/Users/sukesh/Desktop/satquery/backend/data/adaptation/checkpoints"
    metrics_dir: str = "/Users/sukesh/Desktop/satquery/backend/data/adaptation/metrics"
    
    # Reproducibility
    seed: int = 42
    
    def __post_init__(self):
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        
    @property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
