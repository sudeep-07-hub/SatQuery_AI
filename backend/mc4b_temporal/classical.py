"""
classical.py — Non-learned bi-temporal change detection (MC4B classical engine).

Runs on any CPU/MPS machine and operates on the real uploaded pixels:
  * SAR     → log-ratio (difference in dB) with speckle smoothing
  * Optical → Change Vector Analysis (CVA) on per-band standardised reflectance

Thresholding combines Otsu with a physically motivated floor (3 dB for SAR,
2 sigma for optical CVA) so that noise alone is not reported as change.

This is NOT ChangeMamba and never claims to be: every output is labelled with
`source_model = CLASSICAL_CHANGE_DETECTION (<method>)`. It detects *that* and *where*
backscatter/reflectance changed; it does not assign semantic classes
(e.g. "buildings") to the change.
"""
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from raster_io import LoadedRaster
from .localization import mask_to_regions, pixel_to_geo

SAR_DB_THRESHOLD_FLOOR = 3.0      # |Δσ⁰| in dB treated as the minimum meaningful SAR change
OPTICAL_SIGMA_FLOOR = 2.0         # CVA magnitude floor, in standard deviations
MIN_REGION_PIXELS = 20
MAX_REGIONS = 25


def _fill_nan(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    for b in range(out.shape[0]):
        band = out[b]
        finite = np.isfinite(band)
        fill = float(np.median(band[finite])) if finite.any() else 0.0
        band[~finite] = fill
    return out


def _resample_to(src: np.ndarray, height: int, width: int) -> np.ndarray:
    if src.shape[1:] == (height, width):
        return src
    t = torch.from_numpy(src).unsqueeze(0)
    t = F.interpolate(t, size=(height, width), mode="bilinear", align_corners=False)
    return t[0].numpy()


def _box_smooth(arr: np.ndarray, k: int) -> np.ndarray:
    t = torch.from_numpy(arr.astype(np.float32))[None, None]
    t = F.avg_pool2d(F.pad(t, (k // 2,) * 4, mode="replicate"), k, stride=1)
    return t[0, 0].numpy()


def _binary_open(mask: np.ndarray) -> np.ndarray:
    """3x3 morphological opening: removes isolated speckle pixels."""
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    eroded = -F.max_pool2d(-t, 3, stride=1, padding=1)
    opened = F.max_pool2d(eroded, 3, stride=1, padding=1)
    return opened[0, 0].numpy() > 0.5


def otsu_threshold(values: np.ndarray, bins: int = 256):
    """Returns (threshold, separability eta in [0, 1])."""
    values = values[np.isfinite(values)]
    if values.size == 0 or values.max() == values.min():
        return float(values.max() if values.size else 0.0), 0.0
    hist, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    weights = hist.astype(np.float64) / hist.sum()
    w0 = np.cumsum(weights)
    w1 = 1.0 - w0
    mu0_cum = np.cumsum(weights * centers)
    mu_total = mu0_cum[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu_total * w0 - mu0_cum) ** 2 / (w0 * w1)
    between[~np.isfinite(between)] = 0.0
    idx = int(np.argmax(between))
    total_var = float(np.sum(weights * (centers - mu_total) ** 2))
    eta = float(between[idx] / total_var) if total_var > 0 else 0.0
    return float(centers[idx]), max(0.0, min(1.0, eta))


def _looks_like_db(data: np.ndarray) -> bool:
    finite = data[np.isfinite(data)]
    return finite.size > 0 and float(np.median(finite)) < 0.0


def detect_change(before: LoadedRaster, after: LoadedRaster, modality: str) -> Dict:
    """
    Run classical change detection on two co-registered observations.
    `before` defines the output grid; `after` is resampled onto it if sizes differ.
    """
    h, w = before.height, before.width
    b = _fill_nan(before.data)
    a = _fill_nan(_resample_to(after.data, h, w))
    notes = []
    if after.data.shape[1:] != (h, w):
        notes.append(f"After image resampled from {after.data.shape[2]}x{after.data.shape[1]} to {w}x{h} px.")

    if modality == "sar":
        b1, a1 = b[0], a[0]
        if _looks_like_db(before.data) and _looks_like_db(after.data):
            diff = a1 - b1
            method = "SAR log-ratio (dB difference)"
        else:
            eps = 1e-6
            diff = 10.0 * np.log10((np.abs(a1) + eps) / (np.abs(b1) + eps))
            method = "SAR log-ratio (linear intensity)"
        diff = _box_smooth(diff, 5)
        magnitude = np.abs(diff)
        otsu, eta = otsu_threshold(magnitude)
        threshold = max(otsu, SAR_DB_THRESHOLD_FLOOR)
        unit = "dB"
        signed = diff
    else:
        bands = min(b.shape[0], a.shape[0])
        stacked = np.concatenate([b[:bands], a[:bands]], axis=1)
        mean = stacked.reshape(bands, -1).mean(axis=1)[:, None, None]
        std = stacked.reshape(bands, -1).std(axis=1)[:, None, None] + 1e-6
        delta = (a[:bands] - mean) / std - (b[:bands] - mean) / std
        magnitude = _box_smooth(np.sqrt(np.sum(delta ** 2, axis=0)) / np.sqrt(bands), 3)
        m_mean, m_std = float(magnitude.mean()), float(magnitude.std()) + 1e-6
        otsu, eta = otsu_threshold(magnitude)
        threshold = max(otsu, m_mean + OPTICAL_SIGMA_FLOOR * m_std)
        method = f"Optical change vector analysis ({bands} band{'s' if bands != 1 else ''})"
        unit = "sigma"
        signed = np.mean(delta, axis=0)

    mask = _binary_open(magnitude > threshold)

    regions_px = mask_to_regions(torch.from_numpy(mask.astype(np.float32)), min_region_pixels=MIN_REGION_PIXELS)
    regions_px.sort(key=lambda r: r["area_pixels"], reverse=True)

    # Keep only pixels that belong to retained regions
    clean_mask = np.zeros_like(mask)
    for r in regions_px:
        clean_mask |= r["mask_slice"]

    gsd = abs(before.transform[0]) if before.is_georeferenced else None
    regions = []
    for i, r in enumerate(regions_px[:MAX_REGIONS]):
        sl = r["mask_slice"]
        mean_signed = float(signed[sl].mean())
        x0, y0, x1, y1 = r["bbox_pixel"]
        region = {
            "region_index": i,
            "bbox_pixel": r["bbox_pixel"],
            "area_pixels": r["area_pixels"],
            "area_m2": round(r["area_pixels"] * gsd * gsd, 1) if gsd else None,
            "mean_change": round(mean_signed, 3),
            "mean_magnitude": round(float(magnitude[sl].mean()), 3),
            "direction": "increase" if mean_signed > 0 else "decrease",
            "geometry": None,
        }
        if before.is_georeferenced:
            corners = [(x0, y0), (x1 + 1, y0), (x1 + 1, y1 + 1), (x0, y1 + 1), (x0, y0)]
            region["geometry"] = {
                "type": "Polygon",
                "coordinates": [[list(c) for c in pixel_to_geo(corners, before.transform, before.crs)]],
            }
        regions.append(region)

    changed_pixels = int(clean_mask.sum())
    total = int(clean_mask.size)
    stats = {
        "changed_pixels": changed_pixels,
        "total_pixels": total,
        "changed_pixel_pct": round(100.0 * changed_pixels / total, 3) if total else 0.0,
        "changed_area_m2": round(changed_pixels * gsd * gsd, 1) if gsd else None,
        "region_count": len(regions_px),
        "threshold": round(float(threshold), 3),
        "threshold_unit": unit,
        "otsu_threshold": round(float(otsu), 3),
        "threshold_separability": round(eta, 3),
    }
    return {
        "method": method,
        "modality": modality,
        "mask": clean_mask,
        "magnitude": magnitude,
        "threshold": float(threshold),
        "regions": regions,
        "statistics": stats,
        "separability": eta,
        "notes": notes,
    }


def image_bounds_wgs84(raster: LoadedRaster) -> Optional[list]:
    """[[south, west], [north, east]] for Leaflet overlays, or None if not georeferenced."""
    if not raster.is_georeferenced:
        return None
    corners = [(0, 0), (raster.width, 0), (raster.width, raster.height), (0, raster.height)]
    lonlat = pixel_to_geo(corners, raster.transform, raster.crs)
    lons = [p[0] for p in lonlat]
    lats = [p[1] for p in lonlat]
    return [[min(lats), min(lons)], [max(lats), max(lons)]]


def footprint_wgs84(raster: LoadedRaster) -> Optional[dict]:
    if not raster.is_georeferenced:
        return None
    corners = [(0, 0), (raster.width, 0), (raster.width, raster.height), (0, raster.height), (0, 0)]
    return {"type": "Polygon", "coordinates": [[list(c) for c in pixel_to_geo(corners, raster.transform, raster.crs)]]}


def render_change_overlay(result: Dict, path: str) -> str:
    """RGBA PNG: changed pixels in red, opacity scaled by change magnitude."""
    from PIL import Image
    mask = result["mask"]
    mag = result["magnitude"]
    thr = max(result["threshold"], 1e-6)
    alpha = np.clip(mag / (2 * thr), 0.35, 1.0) * 255 * mask
    rgba = np.zeros(mask.shape + (4,), dtype=np.uint8)
    rgba[..., 0] = 239
    rgba[..., 1] = 68
    rgba[..., 2] = 68
    rgba[..., 3] = alpha.astype(np.uint8)
    Image.fromarray(rgba, mode="RGBA").save(path)
    return path
