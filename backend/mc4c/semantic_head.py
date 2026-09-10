import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from configilm.ConfigILM import ILMConfiguration
from .reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
from .reben_publication.BENv2_utils import NEW_LABELS

class SemanticHead:
    def __init__(self, weights_path: str = "mc4c/weights/model.safetensors", k: int = 5):
        # We manually build the configuration for ResNet50
        # Channels: 14 for BENv2 if S1+S2. 
        # Actually BigEarthNetv2_0_ImageClassifier default expects 19 classes
        config = ILMConfiguration(
            timm_model_name="resnet50",
            classes=19,
            channels=12, # 10 S2 + 2 S1 or 12 S2 depending on dataset
            image_size=120,
        )
        self.model = BigEarthNetv2_0_ImageClassifier(config=config)
        
        # Load weights from safetensors
        try:
            state_dict = load_file(weights_path)
            self.model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Warning: could not load semantic head weights from {weights_path}: {e}")
        
        self.model.eval()
        self.k = k
        self.labels = NEW_LABELS

    @torch.no_grad()
    def get_tags(self, images: torch.Tensor) -> list:
        logits = self.model(images)
        probs = torch.sigmoid(logits)
        
        topk_probs, topk_indices = torch.topk(probs, self.k, dim=-1)
        
        batch_tags = []
        for i in range(images.shape[0]):
            tags = [self.labels[idx.item()] for idx in topk_indices[i]]
            batch_tags.append(tags)
            
        return batch_tags
