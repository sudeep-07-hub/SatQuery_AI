# TEST_DATA_MANIFEST.md
## SatQuery AI — E2E Demo Readiness Test Data Manifest
Generated: 2026-09-10T21:35Z

### S1-AAD Root
```
/Users/sukesh/Desktop/satquery/S1-AAD Sentinel-1 Amazon Airstrip Dataset/S1-AAD Sentinel-1 Amazon Airstrip Dataset
```

---

### Single-Image Chips (from `Images_geotiff/`)

| ID     | Path (relative to S1-AAD root)       | Bands | Dtype   | CRS         | Shape     |
|--------|--------------------------------------|-------|---------|-------------|-----------|
| ID_100 | `Images_geotiff/_ID_100.tif`         | 1     | float64 | EPSG:32721  | 201×201   |
| ID_101 | `Images_geotiff/_ID_101.tif`         | 1     | float64 | EPSG:32721  | 200×200   |
| ID_104 | `Images_geotiff/_ID_104.tif`         | 1     | float64 | EPSG:32721  | 200×200   |

All 3 are C-band SAR (Sentinel-1), single-band backscatter. Modality: **SAR**. No optical imagery exists in this dataset.

---

### Bi-Temporal Pairs (from `Change_detection/before/` + `Change_detection/after/`)

| Pair # | Location ID | Before Path                            | After Path                           | CRS        | Same CRS | Notes         |
|--------|-------------|----------------------------------------|--------------------------------------|------------|----------|---------------|
| 1      | ID_100      | `Change_detection/before/_ID_100.tif`  | `Change_detection/after/_ID_100.tif` | EPSG:32721 | ✓        | Near-full overlap |
| 2      | ID_104      | `Change_detection/before/_ID_104.tif`  | `Change_detection/after/_ID_104.tif` | EPSG:32721 | ✓        | Near-full overlap |
| 3      | ID_105      | `Change_detection/before/_ID_105.tif`  | `Change_detection/after/_ID_105.tif` | EPSG:32722 | ✓        | Near-full overlap |

All pairs are 200×200 px, 1-band float64, SAR. Ground truth masks exist in `Change_detection/mask/`.

---

### Dataset Summary
- **Total single-scene GeoTIFFs**: 1,040
- **Total bi-temporal pairs**: 114 (before+after, 113 with ground truth masks)
- **Format**: GeoTIFF, 1-band float64
- **Modality**: SAR (C-band, Sentinel-1) — **no optical imagery**
- **CRS**: EPSG:32721 (majority) and EPSG:32722 (subset)
- **Pixel spacing**: ~10m GSD
