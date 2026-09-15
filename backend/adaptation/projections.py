import torch
import torch.nn as nn
import torch.nn.functional as F

class TextProjectionHead(nn.Module):
    """
    Trainable projection layer mapping Qwen3 hidden states to the shared embedding space.
    """
    def __init__(self, input_dim: int = 2560, output_dim: int = 512):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, input_dim]
        projected = self.projection(x)
        # L2 normalization
        return F.normalize(projected, p=2, dim=-1)

class VisualProjectionHead(nn.Module):
    """
    Trainable projection layer mapping CROMA spatial tokens to the shared embedding space.
    Aggregates spatial tokens using mean pooling before projection.
    """
    def __init__(self, input_dim: int = 768, output_dim: int = 512):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        
    def forward(self, spatial_tokens: torch.Tensor) -> torch.Tensor:
        # spatial_tokens: [B, 225, input_dim]
        # Aggregate spatial tokens using mean pooling over the sequence dimension
        aggregated = spatial_tokens.mean(dim=1)  # -> [B, input_dim]
        projected = self.projection(aggregated)
        # L2 normalization
        return F.normalize(projected, p=2, dim=-1)
