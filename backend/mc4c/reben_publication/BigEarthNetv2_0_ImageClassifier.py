from typing import List, Optional
import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from configilm import ConfigILM
from configilm.ConfigILM import ILMConfiguration
from configilm.ConfigILM import ILMType
from huggingface_hub import PyTorchModelHubMixin

class BigEarthNetv2_0_ImageClassifier(pl.LightningModule, PyTorchModelHubMixin):
    def __init__(self, config: ILMConfiguration, lr: float = 1e-3, warmup: Optional[int] = None):
        super().__init__()
        self.lr = lr
        self.warmup = warmup
        self.config = config
        self.model = ConfigILM.ConfigILM(config)
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, batch):
        return self.model(batch)
