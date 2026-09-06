import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
from PIL import Image

def generate_fixtures():
    fixtures_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Optical-like GeoTIFF (test_geo1.tif)
    # 4 bands (RGBA), EPSG:32643
    # Top left: x=300000, y=4000000. Pixel size: 10m
    geo1_path = os.path.join(fixtures_dir, "test_geo1.tif")
    transform1 = from_origin(300000, 4000000, 10, 10)
    data1 = np.random.randint(0, 255, (4, 100, 100), dtype=np.uint8)
    
    with rasterio.open(
        geo1_path,
        'w',
        driver='GTiff',
        height=100,
        width=100,
        count=4,
        dtype=data1.dtype,
        crs='EPSG:32643',
        transform=transform1,
    ) as dst:
        dst.write(data1)
        # Write some tags to simulate metadata (e.g. date)
        dst.update_tags(TIFFTAG_DATETIME='2024-11-15 10:00:00')

    # 2. SAR-like GeoTIFF (test_geo2.tif)
    # 1 band (float32), EPSG:32643
    # Partially overlaps with geo1
    # geo1 footprint: minx=300000, miny=3999000, maxx=301000, maxy=4000000
    # geo2 top left: x=300500, y=3999500. Pixel size: 10m
    # geo2 footprint: minx=300500, miny=3998500, maxx=301500, maxy=3999500
    # Overlap is from x=300500 to 301000, y=3999000 to 3999500
    geo2_path = os.path.join(fixtures_dir, "test_geo2.tif")
    transform2 = from_origin(300500, 3999500, 10, 10)
    data2 = np.random.rand(1, 100, 100).astype(np.float32)
    
    with rasterio.open(
        geo2_path,
        'w',
        driver='GTiff',
        height=100,
        width=100,
        count=1,
        dtype=data2.dtype,
        crs='EPSG:32643',
        transform=transform2,
    ) as dst:
        dst.write(data2)
        dst.update_tags(TIFFTAG_DATETIME='2024-11-18 10:00:00')

    # 3. Plain PNG without georeferencing
    png_path = os.path.join(fixtures_dir, "test_plain.png")
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    img.save(png_path)

    print(f"Fixtures generated in {fixtures_dir}")

if __name__ == '__main__':
    generate_fixtures()
