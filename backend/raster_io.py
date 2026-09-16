"""
raster_io.py — Loads uploaded observations into arrays/tensors for specialist engines.

Replaces the former placeholder `torch.rand` inputs: every specialist now receives
the pixels of the file the user actually uploaded.
"""
import io
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

UPLOAD_ROOT = os.path.join(os.path.dirname(__file__), "uploads")


@dataclass
class LoadedRaster:
    """Pixel data plus the georeferencing needed to map results back to the ground."""
    observation_id: str
    filename: str
    path: str
    data: np.ndarray                      # (bands, H, W) float32, raw values (NaN = nodata)
    crs: Optional[str] = None
    transform: Optional[List[float]] = None  # [a, b, c, d, e, f] (rasterio Affine order)
    band_descriptions: List[str] = field(default_factory=list)

    @property
    def height(self) -> int:
        return self.data.shape[1]

    @property
    def width(self) -> int:
        return self.data.shape[2]

    @property
    def is_georeferenced(self) -> bool:
        return bool(self.crs) and self.transform is not None


def save_uploads(job_id: str, files_data: List[tuple]) -> List[str]:
    """Persist uploaded bytes to a per-job directory so file-based engines can open them."""
    job_dir = os.path.join(UPLOAD_ROOT, job_id)
    os.makedirs(job_dir, exist_ok=True)
    paths = []
    for idx, (filename, content) in enumerate(files_data):
        safe_name = f"image_{idx + 1}_{os.path.basename(filename)}"
        path = os.path.join(job_dir, safe_name)
        with open(path, "wb") as f:
            f.write(content)
        paths.append(path)
    return paths


def load_raster(path: str, observation_id: str, filename: Optional[str] = None) -> LoadedRaster:
    """Read a GeoTIFF (via rasterio) or a plain PNG/JPEG (via Pillow)."""
    filename = filename or os.path.basename(path)
    ext = os.path.splitext(path)[1].lower()

    if ext in (".tif", ".tiff"):
        import rasterio
        with rasterio.open(path) as src:
            data = src.read(masked=True).astype(np.float32)
            data = np.ma.filled(data, np.nan)
            crs = src.crs.to_string() if src.crs else None
            t = src.transform
            transform = [t.a, t.b, t.c, t.d, t.e, t.f] if crs else None
            descriptions = [d or "" for d in (src.descriptions or [])]
        data[~np.isfinite(data)] = np.nan
        return LoadedRaster(observation_id, filename, path, data, crs, transform, descriptions)

    img = Image.open(path)
    img = img.convert("RGB") if img.mode not in ("L", "RGB") else img
    arr = np.asarray(img, dtype=np.float32)
    arr = arr[np.newaxis, ...] if arr.ndim == 2 else np.transpose(arr, (2, 0, 1))
    return LoadedRaster(observation_id, filename, path, arr)


def _percentile_stretch(band: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    valid = band[np.isfinite(band)]
    if valid.size == 0:
        return np.zeros_like(band, dtype=np.float32)
    lo, hi = np.percentile(valid, [low, high])
    if hi <= lo:
        hi = lo + 1e-6
    out = (np.nan_to_num(band, nan=lo) - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def to_display_rgb(raster: LoadedRaster) -> np.ndarray:
    """(H, W, 3) uint8 visualisation: percentile-stretched RGB or greyscale."""
    data = raster.data
    if data.shape[0] >= 3 and data.shape[0] <= 4:
        bands = [_percentile_stretch(data[i]) for i in range(3)]
    elif data.shape[0] >= 12:
        # Full Sentinel-2 stack starting at B01: true colour = B04, B03, B02
        bands = [_percentile_stretch(data[i]) for i in (3, 2, 1)]
    elif data.shape[0] >= 5:
        # 10-m/20-m Sentinel-2 stack starting at B02 (e.g. BigEarthNet): true colour = B04, B03, B02
        bands = [_percentile_stretch(data[i]) for i in (2, 1, 0)]
    else:
        grey = _percentile_stretch(data[0])
        bands = [grey, grey, grey]
    return (np.stack(bands, axis=-1) * 255).astype(np.uint8)


def to_pil_rgb(raster: LoadedRaster) -> Image.Image:
    return Image.fromarray(to_display_rgb(raster), mode="RGB")


def to_model_tensor(raster: LoadedRaster) -> torch.Tensor:
    """(1, C, H, W) float32 tensor in [0, 1] using a per-band percentile stretch."""
    bands = np.stack([_percentile_stretch(raster.data[i]) for i in range(raster.data.shape[0])])
    return torch.from_numpy(bands).unsqueeze(0)


def preview_png_bytes(raster: LoadedRaster) -> bytes:
    buf = io.BytesIO()
    to_pil_rgb(raster).save(buf, format="PNG")
    return buf.getvalue()


def load_job_rasters(paths: List[str], filenames: List[str]) -> Dict[str, LoadedRaster]:
    """Load every uploaded observation, keyed by the MC1 legacy observation id (image_1, image_2)."""
    rasters = {}
    for idx, (path, filename) in enumerate(zip(paths, filenames)):
        obs_id = f"image_{idx + 1}"
        rasters[obs_id] = load_raster(path, obs_id, filename)
    return rasters
