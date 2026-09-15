import torch
import torch.nn as nn
import torch.nn.functional as F

class SymmetricInfoNCE(nn.Module):
    """
    Symmetric image-text InfoNCE (CLIP-style contrastive objective).
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        # We can make temperature a learnable parameter (nn.Parameter) in advanced setups,
        # but for this specific constraint we use a fixed or manually updated scalar, 
        # or just a tensor to prevent issues. Let's make it a tensor buffer so it's on the right device.
        self.register_buffer('temperature', torch.tensor(temperature))
        
    def forward(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_embeddings: [B, D] L2 normalized
            text_embeddings: [B, D] L2 normalized
        Returns:
            loss: scalar tensor
        """
        batch_size = image_embeddings.size(0)
        
        # If batch size is 1, there are no meaningful negatives.
        # InfoNCE denominator becomes just the positive pair itself.
        # Loss = -log( exp(sim/tau) / exp(sim/tau) ) = -log(1) = 0.0
        # However, CrossEntropyLoss elegantly returns 0.0 without numerical blowup 
        # when num_classes == 1, but some PyTorch versions might warn or error if targets >= num_classes.
        # For batch_size == 1, target is 0, logits is [1, 1]. This is valid.
        
        # Compute cosine similarities
        # image_embeddings: [B, D], text_embeddings.T: [D, B] -> logits: [B, B]
        logits = torch.matmul(image_embeddings, text_embeddings.T) / self.temperature
        
        # The positive pairs are on the diagonal.
        labels = torch.arange(batch_size, device=logits.device, dtype=torch.long)
        
        # Cross entropy loss in both directions
        image_to_text_loss = F.cross_entropy(logits, labels)
        text_to_image_loss = F.cross_entropy(logits.T, labels)
        
        # Symmetric loss
        loss = (image_to_text_loss + text_to_image_loss) / 2.0
        return loss
