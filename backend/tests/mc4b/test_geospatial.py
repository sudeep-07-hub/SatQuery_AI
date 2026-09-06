"""
test_geospatial.py — Geospatial correctness tests.

E. changed_region_coordinates reprojects correctly.
   changed_area_m2 uses actual GSD, not hardcoded pixel count.
"""

import pytest
import torch
import geojson
from mc4b_temporal.tool_adapter import ChangeMambaAdapter
from mc4b_temporal.localization import (
    pixel_to_geo,
    compute_change_statistics,
    mask_to_regions,
    regions_to_geojson,
)


class TestPixelToGeoProjection:
    """Verify pixel→geo reprojection is correct."""

    def test_pixel_to_geo_identity_affine(self):
        """With identity-like affine (GSD=1m, origin at 0,0), pixel ≈ geo."""
        affine = [1.0, 0, 0, 0, -1.0, 100.0]
        pixels = [(10, 20)]
        geo = pixel_to_geo(pixels, affine, "EPSG:32643", "EPSG:32643")
        assert len(geo) == 1
        # x_geo = 1*10 + 0*20 + 0 = 10
        # y_geo = 0*10 + (-1)*20 + 100 = 80
        assert abs(geo[0][0] - 10.0) < 0.01
        assert abs(geo[0][1] - 80.0) < 0.01

    def test_pixel_to_geo_with_offset(self):
        """Affine with UTM origin offset."""
        affine = [10.0, 0, 300000.0, 0, -10.0, 4000000.0]
        pixels = [(0, 0)]
        geo = pixel_to_geo(pixels, affine, "EPSG:32643", "EPSG:32643")
        assert abs(geo[0][0] - 300000.0) < 0.01
        assert abs(geo[0][1] - 4000000.0) < 0.01

    def test_valid_geojson_output(self):
        """regions_to_geojson must produce valid GeoJSON geometries."""
        mask = torch.zeros(1, 1, 64, 64)
        mask[0, 0, 10:20, 10:20] = 1.0  # 10×10 changed block
        regions = mask_to_regions(mask)
        assert len(regions) >= 1

        affine = [1.0, 0, 300000.0, 0, -1.0, 4000000.0]
        geo_regions = regions_to_geojson(regions, affine, "EPSG:32643", 1.0)
        for gr in geo_regions:
            assert "geometry" in gr
            assert "crs" in gr
            assert gr["crs"] == "EPSG:32643"
            # Validate it's a valid GeoJSON geometry
            geom = gr["geometry"]
            assert geom["type"] == "Polygon"
            assert len(geom["coordinates"]) >= 1


class TestAreaComputation:
    """changed_area_m2 must use actual GSD, not a pixel-count assumption."""

    def test_area_uses_gsd(self):
        """100 changed pixels at 10m GSD = 10000 m²."""
        mask = torch.zeros(1, 1, 100, 100)
        mask[0, 0, 0:10, 0:10] = 1.0  # 100 pixels changed
        stats = compute_change_statistics(mask, gsd_m=10.0)
        assert stats["changed_pixels"] == 100
        assert stats["changed_area_m2"] == 10000.0  # 100 * 10^2

    def test_area_with_1m_gsd(self):
        """100 changed pixels at 1m GSD = 100 m²."""
        mask = torch.zeros(1, 1, 100, 100)
        mask[0, 0, 0:10, 0:10] = 1.0
        stats = compute_change_statistics(mask, gsd_m=1.0)
        assert stats["changed_area_m2"] == 100.0

    def test_no_change_area_zero(self):
        """No changed pixels → 0 m²."""
        mask = torch.zeros(1, 1, 64, 64)
        stats = compute_change_statistics(mask, gsd_m=10.0)
        assert stats["changed_area_m2"] == 0.0
        assert stats["changed_pixel_pct"] == 0.0
