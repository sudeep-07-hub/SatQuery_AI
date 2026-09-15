import torch
import torch.nn as nn
import torch.nn.functional as F

class TextProjectionHead(nn.Module):
    """
    Explicit untrained projection interface for the text pathway.
    Maps Qwen pooled hidden states [B, 2560] -> [B, 512].
    """
    def __init__(self, in_dim: int = 2560, out_dim: int = 512):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects and L2-normalizes the text representation.
        """
        x = self.proj(x)
        # L2 normalization safely
        return F.normalize(x, p=2, dim=-1)


class VisualProjectionHead(nn.Module):
    """
    Explicit untrained projection interface for the visual pathway.
    Maps auxiliary global-pooled CROMA features [B, 768] -> [B, 512].
    NOTE: The canonical spatial tokens [225, 768] are retained; this is just an auxiliary branch.
    """
    def __init__(self, in_dim: int = 768, out_dim: int = 512):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects and L2-normalizes the visual representation.
        """
        x = self.proj(x)
        # L2 normalization safely
        return F.normalize(x, p=2, dim=-1)
