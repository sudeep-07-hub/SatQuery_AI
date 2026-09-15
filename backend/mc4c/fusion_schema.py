from typing import Any, Dict, List
from pydantic import BaseModel, Field, model_serializer
import torch

from .token_schema import TokenSpatialIdentity
from .query_schema import QueryRepresentation

class FusedTokenRepresentation(BaseModel):
    """
    Representation of the multimodal spatial tokens after query-conditioned fusion.
    Explicitly preserves the spatial grid geometry (TokenSpatialIdentity).
    """
    grid_height: int
    grid_width: int
    num_tokens: int
    batch_size: int
    
    # Target tensor
    fused_tokens: Any = Field(..., description="Query-conditioned fused tensor (B, N, fusion_dim)")
    fusion_dim: int = Field(default=768, description="Dimensionality of the fused representation")
    
    # Metadata
    spatial_identities: List[TokenSpatialIdentity] = Field(..., description="Explicit spatial identities parallel to sequence dim")
    query_context: QueryRepresentation = Field(..., description="The query representation that conditioned this fusion")
    
    model_config = {"arbitrary_types_allowed": True}

    @model_serializer(mode='wrap')
    def serialize_metadata(self, handler) -> Dict[str, Any]:
        """
        Safely serializes metadata while omitting the raw tensors.
        Prevents exploding audit logs.
        """
        data = handler(self)
        if 'fused_tokens' in data and isinstance(self.fused_tokens, torch.Tensor):
            data['fused_tokens'] = f"<Tensor shape={list(self.fused_tokens.shape)} dtype={str(self.fused_tokens.dtype)}>"
        return data

    def select(self, indices: List[int]) -> 'FusedTokenRepresentation':
        """
        Safely selects a subset of tokens while preserving explicit spatial embeddings.
        Returns a new FusedTokenRepresentation. The original is unchanged.
        """
        if not isinstance(self.fused_tokens, torch.Tensor):
            raise TypeError("Cannot select from non-Tensor fused_tokens")
            
        num_current_tokens = self.fused_tokens.shape[1]
        
        # Verify alignment
        if len(self.spatial_identities) != num_current_tokens:
            raise RuntimeError(f"Metadata misalignment: fused_tokens has {num_current_tokens} tokens but metadata has {len(self.spatial_identities)}")

        # Validate indices
        for idx in indices:
            if not (0 <= idx < num_current_tokens):
                raise IndexError(f"Selection index {idx} out of bounds for token dimension {num_current_tokens}")

        # Subset tensor (B, N, D)
        new_tokens = self.fused_tokens[:, indices, :]

        # Subset identities
        new_identities = [self.spatial_identities[i] for i in indices]

        return FusedTokenRepresentation(
            batch_size=self.batch_size,
            grid_height=self.grid_height,
            grid_width=self.grid_width,
            num_tokens=len(indices),
            fused_tokens=new_tokens,
            fusion_dim=self.fusion_dim,
            spatial_identities=new_identities,
            query_context=self.query_context
        )
