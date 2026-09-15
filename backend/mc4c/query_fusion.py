import torch
import torch.nn as nn

from .embedding_schema import TokenRepresentation
from .query_schema import QueryRepresentation
from .fusion_schema import FusedTokenRepresentation

class QueryConditionedSpatialFusion(nn.Module):
    """
    Lightweight, spatial-preserving multimodal fusion using FiLM-like gating.
    The user query dynamically gates the optical and SAR inputs per spatial token,
    avoiding the massive parameter footprint of a multi-head cross-attention layer
    while mathematically enforcing query-dependent and modality-aware fusion.
    """
    def __init__(
        self,
        token_dim: int = 768,
        query_dim: int = 384,
        modality_dim: int = 16,
        spatial_dim: int = 2,
        fusion_dim: int = 768
    ):
        super().__init__()
        self.fusion_dim = fusion_dim
        
        # Project textual query into fusion space
        self.query_proj = nn.Linear(query_dim, fusion_dim)
        
        # Project concatenated [image_feat, modality_emb, spatial_emb] into fusion space
        in_dim = token_dim + modality_dim + spatial_dim
        self.opt_proj = nn.Linear(in_dim, fusion_dim)
        self.sar_proj = nn.Linear(in_dim, fusion_dim)
        
        # FiLM-like gating network:
        # Input: [query_feat, opt_feat, sar_feat] (3 * fusion_dim)
        # Output: [gate_opt, gate_sar] (2)
        self.gate_net = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, 2),
            nn.Sigmoid()  # Outputs gate weights in (0, 1)
        )
        
        # Initialize randomly (untrained module)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(
        self, 
        query: QueryRepresentation, 
        opt_rep: TokenRepresentation, 
        sar_rep: TokenRepresentation
    ) -> FusedTokenRepresentation:
        
        # Validate metadata alignment
        if opt_rep.num_tokens != sar_rep.num_tokens:
            raise ValueError(f"Token count mismatch: Optical ({opt_rep.num_tokens}) != SAR ({sar_rep.num_tokens})")
            
        if opt_rep.grid_height != sar_rep.grid_height or opt_rep.grid_width != sar_rep.grid_width:
            raise ValueError("Spatial grid mismatch between optical and SAR representations.")
            
        if opt_rep.batch_size != sar_rep.batch_size:
            raise ValueError(f"Batch size mismatch: Optical ({opt_rep.batch_size}) != SAR ({sar_rep.batch_size})")
            
        if query.embedding.shape[0] != opt_rep.batch_size:
            raise ValueError(f"Query batch size ({query.embedding.shape[0]}) != Image batch size ({opt_rep.batch_size})")
            
        # (B, N, D)
        opt_tokens = opt_rep.original_tokens
        opt_mod = opt_rep.modality_embeddings
        opt_spat = opt_rep.spatial_embeddings
        
        sar_tokens = sar_rep.original_tokens
        sar_mod = sar_rep.modality_embeddings
        sar_spat = sar_rep.spatial_embeddings
        
        # Concatenate and project image tokens
        opt_cat = torch.cat([opt_tokens, opt_mod, opt_spat], dim=-1) # (B, N, token_dim+mod_dim+spat_dim)
        sar_cat = torch.cat([sar_tokens, sar_mod, sar_spat], dim=-1) # (B, N, token_dim+mod_dim+spat_dim)
        
        opt_feat = self.opt_proj(opt_cat) # (B, N, fusion_dim)
        sar_feat = self.sar_proj(sar_cat) # (B, N, fusion_dim)
        
        # Project query
        q_feat = self.query_proj(query.embedding) # (B, fusion_dim)
        # Expand query to match spatial sequence length N
        # (B, fusion_dim) -> (B, 1, fusion_dim) -> (B, N, fusion_dim)
        q_feat_expanded = q_feat.unsqueeze(1).expand(-1, opt_feat.shape[1], -1)
        
        # Generate gating weights dynamically based on all three streams
        gate_input = torch.cat([q_feat_expanded, opt_feat, sar_feat], dim=-1) # (B, N, 3*fusion_dim)
        gates = self.gate_net(gate_input) # (B, N, 2)
        
        gate_opt = gates[..., 0:1] # (B, N, 1)
        gate_sar = gates[..., 1:2] # (B, N, 1)
        
        # Apply query-dependent interaction
        fused_tokens = gate_opt * opt_feat + gate_sar * sar_feat # (B, N, fusion_dim)
        
        return FusedTokenRepresentation(
            grid_height=opt_rep.grid_height,
            grid_width=opt_rep.grid_width,
            num_tokens=opt_rep.num_tokens,
            batch_size=opt_rep.batch_size,
            fused_tokens=fused_tokens,
            fusion_dim=self.fusion_dim,
            spatial_identities=opt_rep.spatial_identities,
            query_context=query
        )
