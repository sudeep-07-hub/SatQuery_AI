import torch
import torch.nn as nn
from .token_schema import SpatialTokenGrid
from .embedding_schema import ModalitySpatialConfig, TokenRepresentation

class TokenRepresentationLayer(nn.Module):
    """
    Produces explicit modality and spatial embeddings for a SpatialTokenGrid.
    Preserves the original CROMA token and does not perform fusion.
    """
    def __init__(self, config: ModalitySpatialConfig = None):
        super().__init__()
        self.config = config or ModalitySpatialConfig()
        
        # Untrained, deterministic initialization for modality embeddings
        # 0: optical, 1: sar, 2: joint
        self.modality_embedding = nn.Embedding(3, self.config.modality_dim)
        
        # Deterministic initialization for reproducibility
        # We explicitly preserve the RNG state so we don't mess up global RNG
        rng_state = torch.get_rng_state()
        torch.manual_seed(self.config.seed)
        nn.init.normal_(self.modality_embedding.weight)
        torch.set_rng_state(rng_state)
        
        self.modality_map = {"optical": 0, "sar": 1, "joint": 2}

    def forward(self, grid: SpatialTokenGrid) -> TokenRepresentation:
        """
        Annotates the grid with modality and spatial embeddings.
        Returns a TokenRepresentation.
        """
        b, n, d = grid.tokens.shape
        device = grid.tokens.device
        
        if self.config.spatial_dim != 2:
            raise ValueError(f"Spatial dim must be 2 for [row_norm, col_norm] encoding, got {self.config.spatial_dim}")
            
        # 1. Modality embedding
        mod_idx = self.modality_map[grid.modality]
        # Create a batched sequence of modality indices (B, N)
        mod_indices = torch.full((b, n), mod_idx, dtype=torch.long, device=device)
        mod_emb = self.modality_embedding(mod_indices) # (B, N, modality_dim)
        
        # 2. Spatial embedding (deterministic normalized coordinates)
        # [row_norm, col_norm]
        spat_emb = torch.zeros((b, n, 2), dtype=torch.float32, device=device)
        
        # grid.spatial_identities is length N
        for i, identity in enumerate(grid.spatial_identities):
            # Safe normalization preventing div by zero for 1x1 grids
            row_norm = identity.row / (grid.grid_height - 1) if grid.grid_height > 1 else 0.5
            col_norm = identity.column / (grid.grid_width - 1) if grid.grid_width > 1 else 0.5
            spat_emb[:, i, 0] = row_norm
            spat_emb[:, i, 1] = col_norm
            
        return TokenRepresentation(
            modality=grid.modality,
            grid_height=grid.grid_height,
            grid_width=grid.grid_width,
            num_tokens=grid.num_tokens,
            batch_size=grid.batch_size,
            original_tokens=grid.tokens,
            modality_embeddings=mod_emb,
            spatial_embeddings=spat_emb,
            spatial_identities=grid.spatial_identities
        )
