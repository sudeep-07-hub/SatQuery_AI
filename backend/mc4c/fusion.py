import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from .croma import PretrainedCROMA

class QueryConditionedFusion(nn.Module):
    def __init__(self, croma_weights_path: str, text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        # Load the joint cross-encoder from CROMA
        croma = PretrainedCROMA(pretrained_path=croma_weights_path, size='base', modality='both')
        self.cross_encoder = croma.cross_encoder
        self.attn_bias = croma.attn_bias
        
        # Text embedder
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        
        # Projection from text embed dim (384) to CROMA dim (768)
        self.text_proj = nn.Linear(384, 768)

    def forward(self, sar_features: torch.Tensor, optical_features: torch.Tensor, query: str):
        """
        sar_features: (B, N, D)
        optical_features: (B, N, D)
        query: text string
        """
        device = sar_features.device
        
        # 1. Embed query
        # For simplicity, assuming batch size of 1 for the query or identical query for batch.
        # In a real batch, query would be a list of strings.
        if isinstance(query, str):
            queries = [query] * sar_features.shape[0]
        else:
            queries = query
            
        inputs = self.tokenizer(queries, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            text_outputs = self.text_encoder(**inputs)
        
        # Mean pooling for text embedding
        # shape: (B, 384)
        text_embed = text_outputs.last_hidden_state.mean(dim=1) 
        
        # Project to 768
        text_embed_proj = self.text_proj(text_embed) # (B, 768)
        
        # 2. Condition the features
        # We condition by adding the query embedding to both SAR and optical features before cross attention
        sar_cond = sar_features + text_embed_proj.unsqueeze(1)
        optical_cond = optical_features + text_embed_proj.unsqueeze(1)
        
        # 3. Fuse via CROMA's cross-encoder
        bias = self.attn_bias.to(device)
        joint_encodings = self.cross_encoder(x=sar_cond, context=optical_cond, relative_position_bias=bias)
        
        # Global Average Pooling for final joint feature
        joint_gap = joint_encodings.mean(dim=1) # (B, 768)
        
        return {
            'joint_encodings': joint_encodings.cpu().tolist(),
            'fused_vector': joint_gap.cpu().tolist()
        }
