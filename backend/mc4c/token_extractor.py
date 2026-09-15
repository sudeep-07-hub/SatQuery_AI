import torch
from typing import Dict, Any, Optional
from .token_schema import SpatialTokenGrid, CROMATokenBundle, ModalityType, TokenSpatialIdentity

class CROMATokenExtractor:
    """
    Extracts spatial tokens from CROMA's raw output dictionary into a typed TokenBundle.
    Provides validation without modifying the underlying tensors or CROMA architecture.
    """
    
    def __init__(self, 
                 grid_height: int = 15, 
                 grid_width: int = 15, 
                 token_dim: int = 768, 
                 patch_height: int = 8, 
                 patch_width: int = 8):
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.num_tokens = grid_height * grid_width
        self.token_dim = token_dim
        self.patch_height = patch_height
        self.patch_width = patch_width

    def _extract_modality(self, output_dict: Dict[str, Any], key: str, modality: ModalityType) -> Optional[SpatialTokenGrid]:
        if key not in output_dict:
            return None
            
        tensor = output_dict[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor for {key}, got {type(tensor)}")
            
        if tensor.ndim != 3:
            raise ValueError(f"Expected 3D tensor for {key} (B, N, D), got shape {tensor.shape}")
            
        b, n, d = tensor.shape
        
        if n != self.num_tokens:
            raise ValueError(f"Expected {self.num_tokens} tokens for {key}, got {n}")
            
        if d != self.token_dim:
            raise ValueError(f"Expected token dimension {self.token_dim} for {key}, got {d}")
            
        # Build spatial identities once upon extraction
        identities = []
        for idx in range(self.num_tokens):
            row = idx // self.grid_width
            col = idx % self.grid_width
            min_y = row * self.patch_height
            min_x = col * self.patch_width
            max_y = min_y + self.patch_height
            max_x = min_x + self.patch_width
            identities.append(TokenSpatialIdentity(
                original_index=idx,
                row=row,
                column=col,
                bounds=(min_y, min_x, max_y, max_x)
            ))
            
        return SpatialTokenGrid(
            modality=modality,
            batch_size=b,
            grid_height=self.grid_height,
            grid_width=self.grid_width,
            num_tokens=self.num_tokens,
            token_dim=self.token_dim,
            patch_height=self.patch_height,
            patch_width=self.patch_width,
            tokens=tensor,
            spatial_identities=identities
        )

    def extract(self, croma_output: Dict[str, Any]) -> CROMATokenBundle:
        """
        Extracts available tokens from a CROMA output dictionary.
        Does NOT perform Global Average Pooling.
        """
        optical = self._extract_modality(croma_output, "optical_encodings", "optical")
        sar = self._extract_modality(croma_output, "SAR_encodings", "sar")
        joint = self._extract_modality(croma_output, "joint_encodings", "joint")
        
        return CROMATokenBundle(
            optical=optical,
            sar=sar,
            joint=joint
        )
