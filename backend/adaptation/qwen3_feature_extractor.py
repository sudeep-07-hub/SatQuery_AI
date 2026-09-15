import os
import torch
import torch.nn.functional as F
import psutil
from typing import Dict, Any, Tuple, Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM

class InsufficientMemoryError(Exception):
    pass

def get_optimal_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def masked_mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Applies masked mean pooling to hidden states.
    Args:
        hidden_states: [B, S, D]
        attention_mask: [B, S]
    Returns:
        pooled: [B, D]
    """
    # Expand attention mask to match hidden_states dimensions
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    
    # Sum over the sequence length, clamping the denominator to avoid division by zero
    sum_hidden = (hidden_states * mask).sum(dim=1)
    sum_mask = mask.sum(dim=1).clamp_min(1.0)
    
    return sum_hidden / sum_mask

class Qwen3FeatureExtractor:
    """
    Extracts authentic Qwen3 hidden state representations for given text.
    Enforces strict memory safety and correct layer extraction (Layer 36).
    """
    def __init__(
        self, 
        model_id: str = "Qwen/Qwen3-4B-Instruct-2507", 
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
        max_length: int = 512,
        required_ram_gb: float = 8.0
    ):
        self.model_id = model_id
        self.device = device if device is not None else get_optimal_device()
        self.dtype = dtype
        self.max_length = max_length
        self.required_ram_gb = required_ram_gb
        
        self.tokenizer = None
        self.model = None
        self.model_revision = "unknown"
        self.hidden_dim = 2560
        self.layer = 36

    def check_memory(self):
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)
        if available_gb < self.required_ram_gb:
            raise InsufficientMemoryError(
                f"Insufficient RAM to load Qwen3. Required: {self.required_ram_gb} GB, Available: {available_gb:.2f} GB."
            )

    def load_model(self):
        """
        Loads the tokenizer and the model in inference mode.
        """
        if self.model is not None:
            return
            
        print(f"Checking memory requirements ({self.required_ram_gb} GB needed)...")
        self.check_memory()
        
        print(f"Loading tokenizer for {self.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        # Determine appropriate dtype based on device capabilities
        # Fall back to float32 on MPS since bfloat16 might have limited support depending on macOS version,
        # but try to honor requested dtype if possible.
        actual_dtype = self.dtype
        if self.device.type == 'mps' and actual_dtype == torch.bfloat16:
             print("Warning: Using float32 instead of bfloat16 for MPS compatibility.")
             actual_dtype = torch.float32
             
        print(f"Loading {self.model_id} to {self.device} with dtype {actual_dtype}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=actual_dtype,
            low_cpu_mem_usage=True
        ).to(self.device)
        
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Try to extract the exact revision from the loaded model if available
        if hasattr(self.model.config, '_commit_hash'):
            self.model_revision = self.model.config._commit_hash
            
        print(f"Model loaded successfully. Revision: {self.model_revision}")

    @torch.no_grad()
    def extract(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Extracts features for a batch of texts.
        """
        if self.model is None:
            self.load_model()
            
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=self.max_length
        ).to(self.device)
        
        original_lengths = [len(self.tokenizer.encode(t)) for t in texts]
        truncated = [orig > self.max_length for orig in original_lengths]
        token_counts = [min(orig, self.max_length) for orig in original_lengths]
        
        outputs = self.model(
            **inputs, 
            output_hidden_states=True, 
            return_dict=True
        )
        
        # Get all hidden states. Usually a tuple of length (num_layers + 1)
        hidden_states_tuple = outputs.hidden_states
        
        # Layer 36 is conceptually the last layer for a 36-layer model.
        # hidden_states_tuple[-1] contains the final layer output.
        target_hidden_state = hidden_states_tuple[-1]
        
        if target_hidden_state.shape[-1] != self.hidden_dim:
            print(f"Warning: Expected hidden dim {self.hidden_dim}, got {target_hidden_state.shape[-1]}")
            
        # Apply masked mean pooling
        pooled_embeddings = masked_mean_pooling(target_hidden_state, inputs['attention_mask'])
        
        # Normalize
        normalized_embeddings = F.normalize(pooled_embeddings, p=2, dim=-1)
        
        # Construct result dictionaries
        results = []
        for i in range(len(texts)):
            results.append({
                "pooled_embedding": pooled_embeddings[i].cpu(),
                "normalized_embedding": normalized_embeddings[i].cpu(),
                "token_count": token_counts[i],
                "truncated": truncated[i],
                "layer": self.layer,
                "hidden_dim": self.hidden_dim,
                "model_id": self.model_id,
                "model_revision": self.model_revision,
                "tokenizer_id": self.model_id,
                "pooling": "masked_mean",
                "normalization": "l2",
                "dtype": str(target_hidden_state.dtype),
                "device": str(self.device),
                "max_length": self.max_length
            })
            
        return results
