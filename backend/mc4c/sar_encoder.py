import torch
from .croma import PretrainedCROMA

class SAREncoder:
    def __init__(self, croma_weights_path: str):
        self.model = PretrainedCROMA(pretrained_path=croma_weights_path, size='base', modality='SAR')
        self.model.eval()

    @torch.no_grad()
    def encode(self, sar_images: torch.Tensor):
        """
        Extracts features from SAR images using CROMA.
        sar_images: Tensor of shape (B, 2, 120, 120)
        Returns dict with patch_features and global_feature.
        """
        out = self.model(SAR_images=sar_images)
        return {
            'patch_features': out['SAR_encodings'].cpu().tolist(),
            'global_feature': out['SAR_GAP'].cpu().tolist()
        }
