import torch
from .croma import PretrainedCROMA

class OpticalEncoder:
    def __init__(self, croma_weights_path: str):
        self.model = PretrainedCROMA(pretrained_path=croma_weights_path, size='base', modality='optical')
        self.model.eval()

    @torch.no_grad()
    def encode(self, optical_images: torch.Tensor):
        """
        Extracts features from optical images using CROMA.
        optical_images: Tensor of shape (B, 12, 120, 120)
        Returns dict with patch_features and global_feature.
        """
        out = self.model(optical_images=optical_images)
        return {
            'patch_features': out['optical_encodings'].cpu().tolist(),
            'global_feature': out['optical_GAP'].cpu().tolist()
        }
