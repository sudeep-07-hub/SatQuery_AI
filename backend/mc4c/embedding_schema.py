from typing import Any, Dict
from pydantic import BaseModel, Field, model_serializer
import torch
from .token_schema import ModalityType, TokenSpatialIdentity

class ModalitySpatialConfig(BaseModel):
    """
    Configuration for modality and spatial explicit representations.
    """
    modality_dim: int = 16
    spatial_dim: int = 2  # (row_norm, col_norm)
    seed: int = 42        # Used for deterministic initialization


class TokenRepresentation(BaseModel):
    """
    An explicit annotated representation of a batch of tokens, securely pairing
    the original CROMA tokens with lightweight, separable modality and spatial embeddings.
    """
    modality: ModalityType
    grid_height: int
    grid_width: int
    num_tokens: int
    batch_size: int

    # Tensors
    original_tokens: Any = Field(..., description="Original CROMA token tensor (B, N, D)")
    modality_embeddings: Any = Field(..., description="Modality embedding tensor (B, N, modality_dim)")
    spatial_embeddings: Any = Field(..., description="Spatial embedding tensor (B, N, spatial_dim)")
    
    # Metadata
    spatial_identities: list[TokenSpatialIdentity] = Field(..., description="Explicit spatial identities parallel to sequence dim")

    model_config = {"arbitrary_types_allowed": True}

    @model_serializer(mode='wrap')
    def serialize_metadata(self, handler) -> Dict[str, Any]:
        """
        Safely serializes metadata while omitting the raw tensors.
        Prevents exploding audit logs with giant float arrays.
        """
        data = handler(self)
        for key in ["original_tokens", "modality_embeddings", "spatial_embeddings"]:
            if key in data and isinstance(getattr(self, key), torch.Tensor):
                tensor = getattr(self, key)
                data[key] = f"<Tensor shape={list(tensor.shape)} dtype={str(tensor.dtype)}>"
        return data

    def select(self, indices: list[int]) -> 'TokenRepresentation':
        """
        Safely selects a subset of tokens while preserving their aligned explicit modality and spatial embeddings.
        Returns a new TokenRepresentation. The original is unchanged.
        """
        if not isinstance(self.original_tokens, torch.Tensor):
            raise TypeError("Cannot select from non-Tensor original_tokens")
            
        num_current_tokens = self.original_tokens.shape[1]
        
        # Verify alignment
        if self.modality_embeddings.shape[1] != num_current_tokens:
            raise RuntimeError(f"Metadata misalignment: original_tokens has {num_current_tokens} tokens but modality_embeddings has {self.modality_embeddings.shape[1]}")
        if self.spatial_embeddings.shape[1] != num_current_tokens:
            raise RuntimeError(f"Metadata misalignment: original_tokens has {num_current_tokens} tokens but spatial_embeddings has {self.spatial_embeddings.shape[1]}")
        if len(self.spatial_identities) != num_current_tokens:
            raise RuntimeError(f"Metadata misalignment: original_tokens has {num_current_tokens} tokens but metadata has {len(self.spatial_identities)}")

        # Validate indices
        for idx in indices:
            if not (0 <= idx < num_current_tokens):
                raise IndexError(f"Selection index {idx} out of bounds for token dimension {num_current_tokens}")

        # Subset tensors (B, N, D)
        new_tokens = self.original_tokens[:, indices, :]
        new_modality = self.modality_embeddings[:, indices, :]
        new_spatial = self.spatial_embeddings[:, indices, :]

        # Subset identities
        new_identities = [self.spatial_identities[i] for i in indices]

        return TokenRepresentation(
            modality=self.modality,
            batch_size=self.batch_size,
            grid_height=self.grid_height,
            grid_width=self.grid_width,
            num_tokens=len(indices),
            original_tokens=new_tokens,
            modality_embeddings=new_modality,
            spatial_embeddings=new_spatial,
            spatial_identities=new_identities
        )
