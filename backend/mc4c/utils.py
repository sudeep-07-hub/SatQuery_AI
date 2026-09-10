import rasterio
import torch
import numpy as np
from rasterio.enums import Resampling
import torchvision.transforms.functional as F

def load_and_preprocess_image(file_path: str, expected_channels: int, target_size: int = 120) -> torch.Tensor:
    """
    Loads a GeoTIFF image using rasterio, resamples to target_size x target_size,
    and returns a PyTorch tensor of shape (1, expected_channels, target_size, target_size).
    """
    with rasterio.open(file_path) as src:
        # Read all bands
        data = src.read(
            out_shape=(src.count, target_size, target_size),
            resampling=Resampling.bilinear
        )
        
        # Ensure we have the correct number of channels
        if data.shape[0] > expected_channels:
            data = data[:expected_channels, :, :]
        elif data.shape[0] < expected_channels:
            # Pad with zeros if fewer channels
            pad = np.zeros((expected_channels - data.shape[0], target_size, target_size), dtype=data.dtype)
            data = np.concatenate([data, pad], axis=0)

        # Convert to float32 and normalize roughly (0-1) assuming uint16 input
        # Note: In a production system, use BigEarthNet specific mean/std per band.
        data = data.astype(np.float32)
        if data.max() > 1.0:
            data = data / 10000.0  # typical Sentinel-2 scaling

        tensor = torch.from_numpy(data).unsqueeze(0)  # Add batch dimension
        return tensor
