from typing import Literal, Tuple, Dict, Any, Optional
from pydantic import BaseModel, Field, model_serializer
import torch

ModalityType = Literal["optical", "sar", "joint"]

class TokenSpatialIdentity(BaseModel):
    """
    Explicit, immutable spatial identity for a single token.
    Ensures spatial information survives subsetting or reordering.
    """
    original_index: int
    row: int
    column: int
    bounds: Tuple[int, int, int, int]  # (min_y, min_x, max_y, max_x)

class SpatialTokenGrid(BaseModel):
    """
    Strongly typed representation of spatial tokens extracted from CROMA.
    Preserves modality, spatial grid mapping, and the raw tensor without pooling.
    """
    modality: ModalityType
    
    # Grid metadata
    batch_size: int = Field(default=1)
    grid_height: int = Field(default=15)
    grid_width: int = Field(default=15)
    num_tokens: int = Field(default=225)
    token_dim: int = Field(default=768)
    patch_height: int = Field(default=8)
    patch_width: int = Field(default=8)
    
    # The raw tokens tensor. Shape must be (B, num_tokens, token_dim)
    tokens: Any = Field(..., description="Raw token tensor (B, N, D)")
    
    # Explicit spatial identities matching the tensor's N dimension
    spatial_identities: list[TokenSpatialIdentity] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

        
    @model_serializer(mode='wrap')
    def serialize_metadata(self, handler) -> Dict[str, Any]:
        """
        Safely serializes metadata while omitting the raw tensor.
        Prevents exploding audit logs with giant float arrays.
        """
        data = handler(self)
        if 'tokens' in data and isinstance(self.tokens, torch.Tensor):
            data['tokens'] = f"<Tensor shape={list(self.tokens.shape)} dtype={str(self.tokens.dtype)}>"
        return data
        
    def index_to_coord(self, index: int) -> Tuple[int, int]:
        """
        Maps a 1D token index to a 2D (row, col) grid coordinate.
        """
        if not (0 <= index < self.num_tokens):
            raise ValueError(f"Index {index} out of bounds for {self.num_tokens} tokens")
        row = index // self.grid_width
        col = index % self.grid_width
        return (row, col)
        
    def coord_to_index(self, row: int, col: int) -> int:
        """
        Maps a 2D (row, col) grid coordinate to a 1D token index.
        """
        if not (0 <= row < self.grid_height) or not (0 <= col < self.grid_width):
            raise ValueError(f"Coordinate ({row}, {col}) out of bounds for {self.grid_height}x{self.grid_width} grid")
        return row * self.grid_width + col
        
    def patch_bounds(self, index: int) -> Tuple[int, int, int, int]:
        """
        Returns the pixel-space bounds for a given token index.
        Returns: (min_y, min_x, max_y, max_x)
        """
        row, col = self.index_to_coord(index)
        min_y = row * self.patch_height
        min_x = col * self.patch_width
        max_y = min_y + self.patch_height
        max_x = min_x + self.patch_width
        return (min_y, min_x, max_y, max_x)

    def select(self, indices: list[int]) -> 'SpatialTokenGrid':
        """
        Safely selects a subset of tokens while preserving their original spatial identities.
        Returns a new SpatialTokenGrid. The original is unchanged.
        """
        if not isinstance(self.tokens, torch.Tensor):
            raise TypeError("Cannot select from non-Tensor tokens")
            
        if self.tokens.shape[1] != len(self.spatial_identities):
            raise RuntimeError(f"Metadata misalignment: tensor has {self.tokens.shape[1]} tokens but metadata has {len(self.spatial_identities)}")
            
        # Validate indices
        num_current_tokens = self.tokens.shape[1]
        for idx in indices:
            if not (0 <= idx < num_current_tokens):
                raise IndexError(f"Selection index {idx} out of bounds for token dimension {num_current_tokens}")
                
        # Subset tensor (B, N, D)
        new_tokens = self.tokens[:, indices, :]
        
        # Subset identities
        new_identities = [self.spatial_identities[i] for i in indices]
        
        # We preserve the overall grid context (grid_height, grid_width, etc.)
        # so downstream components know the original image space this came from.
        # But num_tokens is updated to reflect the current tensor size.
        return SpatialTokenGrid(
            modality=self.modality,
            batch_size=self.batch_size,
            grid_height=self.grid_height,
            grid_width=self.grid_width,
            num_tokens=len(indices),
            token_dim=self.token_dim,
            patch_height=self.patch_height,
            patch_width=self.patch_width,
            tokens=new_tokens,
            spatial_identities=new_identities
        )



class CROMATokenBundle(BaseModel):
    """
    A collection of spatial token grids extracted from a single CROMA forward pass.
    """
    optical: Optional[SpatialTokenGrid] = None
    sar: Optional[SpatialTokenGrid] = None
    joint: Optional[SpatialTokenGrid] = None
