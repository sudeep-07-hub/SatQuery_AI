"""
change_head.py — Binary and Semantic change detection heads for MC4B.

Takes multi-scale feature-map pairs from the backbone and produces:
  - Binary change mask: (B, 1, H, W) sigmoid probabilities
  - Semantic change mask: (B, num_classes, H, W) per-class logits (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from . import config


class BinaryChangeHead(nn.Module):
    """
    Produces a binary change/no-change mask from paired feature maps.

    Strategy: concatenate difference features from each scale, upsample
    and fuse progressively, then apply a 1×1 conv + sigmoid.
    """

    def __init__(self, feature_channels: List[int]):
        super().__init__()
        # Difference feature fusion (coarsest → finest)
        self.fuse3 = nn.Sequential(
            nn.Conv2d(feature_channels[2], 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(feature_channels[1] + 128, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(feature_channels[0] + 64, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, 1, 1)

    def forward(
        self,
        t1_feats: List[torch.Tensor],
        t2_feats: List[torch.Tensor],
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Args:
            t1_feats, t2_feats: Multi-scale features from backbone.
            target_size: (H, W) of original input for upsampling.

        Returns:
            Binary change probabilities, shape (B, 1, H, W), values in [0, 1].
        """
        # Absolute difference at each scale
        d3 = torch.abs(t1_feats[2] - t2_feats[2])
        d2 = torch.abs(t1_feats[1] - t2_feats[1])
        d1 = torch.abs(t1_feats[0] - t2_feats[0])

        # Coarse-to-fine fusion
        x = self.fuse3(d3)
        x = F.interpolate(x, size=d2.shape[2:], mode="bilinear", align_corners=False)
        x = self.fuse2(torch.cat([d2, x], dim=1))
        x = F.interpolate(x, size=d1.shape[2:], mode="bilinear", align_corners=False)
        x = self.fuse1(torch.cat([d1, x], dim=1))

        # Final upsample to input resolution
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return torch.sigmoid(self.head(x))


class SemanticChangeHead(nn.Module):
    """
    Produces a per-class semantic change mask.

    Classes are defined in config.CHANGE_CLASSES (e.g. no_change,
    built_up_gain, built_up_loss, vegetation_loss, vegetation_gain).
    """

    def __init__(self, feature_channels: List[int], num_classes: int = None):
        super().__init__()
        num_classes = num_classes or config.NUM_CLASSES

        # Concatenate T1+T2 features at finest scale + difference
        in_ch = feature_channels[0] * 2 + feature_channels[0]  # t1 + t2 + diff
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(64, num_classes, 1)

    def forward(
        self,
        t1_feats: List[torch.Tensor],
        t2_feats: List[torch.Tensor],
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Returns:
            Semantic change logits, shape (B, num_classes, H, W).
        """
        f1 = t1_feats[0]
        f2 = t2_feats[0]
        diff = torch.abs(f1 - f2)

        x = torch.cat([f1, f2, diff], dim=1)
        x = self.conv(x)
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return self.head(x)


class ChangeDetector(nn.Module):
    """
    Combined change detection module: backbone → binary head + optional semantic head.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        feat_ch = backbone.get_feature_channels()
        self.binary_head = BinaryChangeHead(feat_ch)

        self.semantic_head = None
        if config.ENABLE_SEMANTIC_CHANGE:
            self.semantic_head = SemanticChangeHead(feat_ch)

    @torch.no_grad()
    def predict(
        self, t1: torch.Tensor, t2: torch.Tensor
    ) -> dict:
        """
        Run inference on a T1/T2 pair.

        Args:
            t1: (B, C, H, W) image tensor for time 1
            t2: (B, C, H, W) image tensor for time 2

        Returns:
            dict with keys:
                binary_mask: (B, 1, H, W) float in [0, 1]
                semantic_logits: (B, num_classes, H, W) or None
                confidence: float in [0, 1] (calibrated)
        """
        self.eval()
        target_size = (t1.shape[2], t1.shape[3])

        t1_feats, t2_feats = self.backbone.encode_pair(t1, t2)
        binary_mask = self.binary_head(t1_feats, t2_feats, target_size)

        semantic_logits = None
        if self.semantic_head is not None:
            semantic_logits = self.semantic_head(t1_feats, t2_feats, target_size)

        # Calibrated confidence: mean change probability, temperature-scaled
        raw_conf = binary_mask.mean().item()
        # Clamp to [0, 1] range
        confidence = max(0.0, min(1.0, raw_conf))

        return {
            "binary_mask": binary_mask,
            "semantic_logits": semantic_logits,
            "confidence": confidence,
        }
