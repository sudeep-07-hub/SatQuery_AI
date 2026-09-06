"""
localization.py — Change mask → georeferenced coordinates for MC4B.

Converts binary change masks from pixel space into:
  - Bounding boxes (pixel + georeferenced)
  - Polygons (pixel + georeferenced GeoJSON)
  - Area statistics using actual GSD

Uses the CRS and affine transform from the MC1 Structured Input Profile
to reproject pixel coordinates into the source coordinate system.
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from pyproj import Transformer
import geojson


def mask_to_regions(
    binary_mask: torch.Tensor,
    threshold: float = 0.5,
    min_region_pixels: int = 10,
) -> List[Dict]:
    """
    Extract connected regions from a binary change mask.

    Uses a simple flood-fill approach (no OpenCV dependency).

    Args:
        binary_mask: (1, 1, H, W) or (H, W) float tensor, values in [0, 1].
        threshold: Probability cutoff for change.
        min_region_pixels: Ignore regions smaller than this.

    Returns:
        List of dicts, each with:
            bbox_pixel: [x_min, y_min, x_max, y_max] in pixel coords
            centroid_pixel: [cx, cy]
            area_pixels: int
            mask_slice: np.ndarray of the region mask
    """
    if binary_mask.dim() == 4:
        mask_np = binary_mask[0, 0].cpu().numpy()
    elif binary_mask.dim() == 2:
        mask_np = binary_mask.cpu().numpy()
    else:
        mask_np = binary_mask.squeeze().cpu().numpy()

    binary = (mask_np > threshold).astype(np.uint8)

    # Simple connected-component labeling via iterative flood-fill
    labels = np.zeros_like(binary, dtype=np.int32)
    current_label = 0
    h, w = binary.shape

    for y in range(h):
        for x in range(w):
            if binary[y, x] == 1 and labels[y, x] == 0:
                current_label += 1
                # BFS flood fill
                stack = [(y, x)]
                while stack:
                    cy, cx = stack.pop()
                    if (
                        0 <= cy < h
                        and 0 <= cx < w
                        and binary[cy, cx] == 1
                        and labels[cy, cx] == 0
                    ):
                        labels[cy, cx] = current_label
                        stack.extend(
                            [(cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)]
                        )

    regions = []
    for lbl in range(1, current_label + 1):
        region_mask = labels == lbl
        area = int(region_mask.sum())
        if area < min_region_pixels:
            continue

        ys, xs = np.where(region_mask)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        centroid = [float(xs.mean()), float(ys.mean())]

        regions.append(
            {
                "bbox_pixel": bbox,
                "centroid_pixel": centroid,
                "area_pixels": area,
                "mask_slice": region_mask,
            }
        )

    return regions


def pixel_to_geo(
    pixel_coords: List[Tuple[float, float]],
    affine_transform: List[float],
    source_crs: str,
    target_crs: str = "EPSG:4326",
) -> List[Tuple[float, float]]:
    """
    Convert pixel coordinates to georeferenced coordinates.

    Args:
        pixel_coords: List of (x_pixel, y_pixel) tuples.
        affine_transform: 6-element affine [a, b, c, d, e, f] where:
            x_geo = a * x_pixel + b * y_pixel + c
            y_geo = d * x_pixel + e * y_pixel + f
        source_crs: CRS of the affine transform (e.g. "EPSG:32643").
        target_crs: Target CRS for output (default WGS84).

    Returns:
        List of (lon, lat) tuples in the target CRS.
    """
    a, b, c, d, e, f = affine_transform

    # Apply affine transform to get projected coordinates
    projected = []
    for px, py in pixel_coords:
        x_geo = a * px + b * py + c
        y_geo = d * px + e * py + f
        projected.append((x_geo, y_geo))

    # Reproject from source CRS to target CRS
    if source_crs != target_crs:
        transformer = Transformer.from_crs(
            source_crs, target_crs, always_xy=True
        )
        reprojected = [transformer.transform(x, y) for x, y in projected]
        return reprojected

    return projected


def regions_to_geojson(
    regions: List[Dict],
    affine_transform: List[float],
    source_crs: str,
    gsd_m: float,
) -> List[Dict]:
    """
    Convert pixel-space regions to georeferenced GeoJSON features.

    Args:
        regions: Output of mask_to_regions().
        affine_transform: 6-element affine from MC1 metadata.
        source_crs: Source CRS string (e.g. "EPSG:32643").
        gsd_m: Ground Sampling Distance in meters (for area calculation).

    Returns:
        List of dicts with:
            geometry: GeoJSON geometry object (polygon bbox)
            crs: source CRS string
            area_m2: area in square meters (using actual GSD)
            bbox_geo: georeferenced bounding box
    """
    geo_regions = []

    for region in regions:
        bbox = region["bbox_pixel"]
        x_min, y_min, x_max, y_max = bbox

        # Convert bbox corners to geo coordinates
        corners_pixel = [
            (x_min, y_min),
            (x_max, y_min),
            (x_max, y_max),
            (x_min, y_max),
            (x_min, y_min),  # close the polygon
        ]
        corners_geo = pixel_to_geo(corners_pixel, affine_transform, source_crs)

        # Create GeoJSON polygon
        polygon = geojson.Polygon([corners_geo])
        feature = geojson.Feature(geometry=polygon)

        # Compute area using actual GSD (not hardcoded pixel assumption)
        area_m2 = region["area_pixels"] * (gsd_m ** 2)

        # Georeferenced bbox
        geo_xs = [c[0] for c in corners_geo[:4]]
        geo_ys = [c[1] for c in corners_geo[:4]]
        bbox_geo = [min(geo_xs), min(geo_ys), max(geo_xs), max(geo_ys)]

        geo_regions.append(
            {
                "geometry": feature["geometry"],
                "crs": "EPSG:4326",
                "area_m2": round(area_m2, 2),
                "bbox_geo": bbox_geo,
            }
        )

    return geo_regions


def compute_change_statistics(
    binary_mask: torch.Tensor,
    gsd_m: float,
    threshold: float = 0.5,
) -> Dict:
    """
    Compute aggregate change statistics.

    Args:
        binary_mask: (B, 1, H, W) or (H, W) change probability mask.
        gsd_m: Ground sampling distance in meters.
        threshold: Change probability threshold.

    Returns:
        dict with changed_area_m2, changed_pixel_pct, total_pixels,
        changed_pixels.
    """
    if binary_mask.dim() == 4:
        mask = binary_mask[0, 0]
    elif binary_mask.dim() == 2:
        mask = binary_mask
    else:
        mask = binary_mask.squeeze()

    total_pixels = int(mask.numel())
    changed_pixels = int((mask > threshold).sum().item())
    changed_pct = (changed_pixels / total_pixels * 100) if total_pixels > 0 else 0.0
    changed_area_m2 = changed_pixels * (gsd_m ** 2)

    return {
        "changed_area_m2": round(changed_area_m2, 2),
        "changed_pixel_pct": round(changed_pct, 4),
        "total_pixels": total_pixels,
        "changed_pixels": changed_pixels,
    }
