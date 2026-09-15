from typing import Any, Dict, List
from pydantic import BaseModel, Field, model_serializer
import torch

from .token_schema import TokenSpatialIdentity
from .query_schema import QueryRepresentation

class QwenVisualRepresentation(BaseModel):
    """
    Representation of the visual tokens projected into the Qwen LLM embedding space.
    Strictly preserves the spatial grid geometry across the sequence.
    """
    grid_height: int
    grid_width: int
    num_tokens: int
    batch_size: int
    
    # Target tensor
    projected_tokens: Any = Field(..., description="Projected tensor mapped into Qwen hidden space (B, N, embedding_dim)")
    embedding_dim: int = Field(default=2560, description="Dimensionality of the Qwen3 4B embedding space")
    
    # Metadata preserved
    spatial_identities: List[TokenSpatialIdentity] = Field(..., description="Explicit spatial identities parallel to sequence dim")
    query_context: QueryRepresentation = Field(..., description="The query representation that initially conditioned this stream")
    
    # Provenance
    adapter_type: str = Field(default="QwenProjectionAdapter", description="Component that generated this tensor")
    
    model_config = {"arbitrary_types_allowed": True}

    @model_serializer(mode='wrap')
    def serialize_metadata(self, handler) -> Dict[str, Any]:
        """
        Safely serializes metadata while omitting the raw projected tensors.
        Prevents exploding audit logs.
        """
        data = handler(self)
        if 'projected_tokens' in data and isinstance(self.projected_tokens, torch.Tensor):
            data['projected_tokens'] = f"<Tensor shape={list(self.projected_tokens.shape)} dtype={str(self.projected_tokens.dtype)}>"
        return data
