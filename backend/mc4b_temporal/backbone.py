"""
backbone.py — Bi-Temporal Encoder backbone for MC4B.

Provides BackboneBase (ABC) and LightweightCNNBackbone (CPU stand-in).

Architecture note:
    The real ChangeMamba backbone uses Visual State-Space (VSS/Mamba) blocks
    for long-range spatial context at sub-quadratic cost. The lightweight CNN
    stand-in uses simple Conv2d + BatchNorm + ReLU blocks to produce
    structurally identical multi-scale feature maps, enabling full agentic
    pipeline testing without CUDA.

    To swap in ChangeMamba:
        1. Install mamba-ssm (requires CUDA)
        2. Implement ChangeMambaBackbone(BackboneBase)
        3. Set config.BACKBONE = "changemamba"
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Tuple


class BackboneBase(ABC):
    """
    Abstract base class for bi-temporal encoder backbones.
    Any backbone (ChangeMamba, CNN, ViT, etc.) must implement this interface.
    """

    @abstractmethod
    def encode_pair(
        self, t1: torch.Tensor, t2: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Encode a bi-temporal image pair through shared-weight backbone.

        Args:
            t1: Image at time 1, shape (B, C, H, W)
            t2: Image at time 2, shape (B, C, H, W)

        Returns:
            Tuple of (t1_features, t2_features) where each is a list of
            multi-scale feature maps at different resolutions:
                [scale_1 (B, F1, H/4, W/4),
                 scale_2 (B, F2, H/8, W/8),
                 scale_3 (B, F3, H/16, W/16)]
        """
        ...

    @abstractmethod
    def get_feature_channels(self) -> List[int]:
        """Return the channel count at each feature scale."""
        ...


class _ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU helper."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LightweightCNNBackbone(nn.Module, BackboneBase):
    """
    CPU-compatible Siamese CNN backbone (stand-in for ChangeMamba).

    Shared-weight encoder producing 3 scales of feature maps.
    This is NOT the final production backbone — it exists so that
    all agentic contracts can be tested end-to-end without CUDA.
    """

    FEATURE_CHANNELS = [64, 128, 256]

    def __init__(self, in_channels: int = 3):
        super().__init__()
        # Shared encoder (same weights for T1 and T2)
        self.stage1 = nn.Sequential(
            _ConvBlock(in_channels, 64, stride=2),  # H/2
            _ConvBlock(64, 64, stride=2),  # H/4
        )
        self.stage2 = _ConvBlock(64, 128, stride=2)  # H/8
        self.stage3 = _ConvBlock(128, 256, stride=2)  # H/16

    def _encode_single(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Run one image through the shared encoder."""
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        return [s1, s2, s3]

    def encode_pair(
        self, t1: torch.Tensor, t2: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Encode T1 and T2 through shared weights."""
        return self._encode_single(t1), self._encode_single(t2)

    def get_feature_channels(self) -> List[int]:
        return self.FEATURE_CHANNELS


def get_backbone(name: str = "lightweight_cnn", in_channels: int = 3) -> nn.Module:
    """Factory function for backbone selection."""
    if name == "lightweight_cnn":
        return LightweightCNNBackbone(in_channels=in_channels)
    elif name == "changemamba":
        raise NotImplementedError(
            "ChangeMamba backbone requires CUDA + mamba-ssm. "
            "Install mamba-ssm and implement ChangeMambaBackbone."
        )
    else:
        raise ValueError(f"Unknown backbone: {name}")
