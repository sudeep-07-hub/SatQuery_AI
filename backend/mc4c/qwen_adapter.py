import torch
import torch.nn as nn

from .fusion_schema import FusedTokenRepresentation
from .qwen_adapter_schema import QwenVisualRepresentation

class QwenProjectionAdapter(nn.Module):
    """
    Adapter that projects CROMA visual fusion representations into the explicitly configured
    Qwen3 4B embedding space. Preserves spatial boundaries strictly.
    """
    def __init__(self, input_dim: int = 768, qwen_hidden_dim: int = 2560):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = qwen_hidden_dim
        
        # A simple trainable linear layer to bridge the representation dimensions
        self.projection = nn.Linear(input_dim, qwen_hidden_dim)
        
        # Random initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, fused_rep: FusedTokenRepresentation) -> QwenVisualRepresentation:
        """
        Projects the fused token sequences into Qwen space.
        """
        # Validate inputs explicitly
        if not isinstance(fused_rep, FusedTokenRepresentation):
            raise TypeError("Expected FusedTokenRepresentation")
            
        fused_tensor = fused_rep.fused_tokens
        
        if len(fused_tensor.shape) != 3:
            raise ValueError(f"Expected 3D tensor (B, N, D), got rank {len(fused_tensor.shape)}")
            
        b, n, d = fused_tensor.shape
        
        if d != self.input_dim:
            raise ValueError(f"Expected feature dimension {self.input_dim}, got {d}")
            
        if len(fused_rep.spatial_identities) != n:
            raise ValueError(f"Metadata alignment failed: {len(fused_rep.spatial_identities)} identities for {n} tokens")

        # Project features
        # (B, N, 768) -> (B, N, 2560)
        projected = self.projection(fused_tensor)

        return QwenVisualRepresentation(
            grid_height=fused_rep.grid_height,
            grid_width=fused_rep.grid_width,
            num_tokens=fused_rep.num_tokens,
            batch_size=fused_rep.batch_size,
            projected_tokens=projected,
            embedding_dim=self.output_dim,
            spatial_identities=fused_rep.spatial_identities,
            query_context=fused_rep.query_context
        )
