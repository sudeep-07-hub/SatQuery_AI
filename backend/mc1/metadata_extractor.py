import io
import rasterio
from rasterio.enums import Resampling
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS

def extract_metadata(file_bytes: bytes, detected_format: str) -> dict:
    """
    Extracts metadata based on the detected format.
    Returns a dictionary of metadata.
    """
    meta = {
        "width": None,
        "height": None,
        "crs": None,
        "transform": None,
        "bounds": None,
        "band_count": None,
        "dtypes": None,
        "acquisition_date": None,
        "gsd_m": None,
        "descriptions": [],
        "colorinterp": [],
        "tags": {},
        "image_statistics": {}
    }
    
    if detected_format == "tiff":
        try:
            with rasterio.MemoryFile(file_bytes) as memfile:
                with memfile.open() as dataset:
                    meta["width"] = dataset.width
                    meta["height"] = dataset.height
                    
                    if dataset.crs:
                        meta["crs"] = dataset.crs.to_string()
                    
                    meta["transform"] = dataset.transform
                    meta["bounds"] = dataset.bounds
                    meta["band_count"] = dataset.count
                    meta["dtypes"] = dataset.dtypes
                    
                    # New metadata fields
                    meta["descriptions"] = list(dataset.descriptions) if dataset.descriptions else []
                    meta["colorinterp"] = [ci.name for ci in dataset.colorinterp] if dataset.colorinterp else []
                    meta["tags"] = dataset.tags() or {}
                    
                    # Try to extract date from tags
                    if 'TIFFTAG_DATETIME' in meta["tags"]:
                        meta["acquisition_date"] = meta["tags"]['TIFFTAG_DATETIME']
                        
                    # Calculate GSD (assume transform scale gives pixel size)
                    if dataset.transform:
                        meta["gsd_m"] = abs(dataset.transform.a) # Pixel width
                        
                    # Deterministic downsampled read for image statistics (~512x512)
                    if dataset.width > 0 and dataset.height > 0:
                        scale_x = max(1, dataset.width // 512)
                        scale_y = max(1, dataset.height // 512)
                        out_shape = (dataset.count, int(dataset.height / scale_y), int(dataset.width / scale_x))
                        
                        if out_shape[1] > 0 and out_shape[2] > 0:
                            data = dataset.read(out_shape=out_shape, resampling=Resampling.nearest)
                            
                            # Use Rasterio's robust mask logic
                            masks = dataset.read_masks(out_shape=out_shape, resampling=Resampling.nearest)
                            valid_pixel_mask = np.all(masks == 255, axis=0)
                                
                            if valid_pixel_mask.any():
                                valid_data = data[:, valid_pixel_mask]
                                meta["image_statistics"] = {
                                    "std": float(np.std(valid_data)),
                                    "mean": float(np.mean(valid_data)),
                                    "valid_fraction": float(np.sum(valid_pixel_mask) / valid_pixel_mask.size)
                                }
                                
                    # Extract cloud metadata if present
                    for k, v in meta["tags"].items():
                        kl = k.lower()
                        if "cloud_cover" in kl or kl == "cloud_coverage_assessment":
                            try:
                                meta["cloud_fraction"] = float(v) / 100.0
                                break
                            except (ValueError, TypeError):
                                pass

        except Exception as e:
            # Add warning later
            pass
            
    elif detected_format in ["png", "jpeg"]:
        try:
            img = Image.open(io.BytesIO(file_bytes))
            meta["width"] = img.width
            meta["height"] = img.height
            meta["band_count"] = len(img.getbands())
            meta["colorinterp"] = img.getbands()
            
            # Extract basic stats using numpy
            data = np.array(img)
            meta["image_statistics"] = {
                "std": float(np.std(data)),
                "mean": float(np.mean(data)),
                "valid_fraction": 1.0 # Standard images typically don't have nodata masks
            }

            
            # Try EXIF
            if hasattr(img, '_getexif') and img._getexif():
                exif = {TAGS.get(k, k): v for k, v in img._getexif().items()}
                meta["tags"] = exif
                if 'DateTimeOriginal' in exif:
                    meta["acquisition_date"] = exif['DateTimeOriginal']
                elif 'DateTime' in exif:
                    meta["acquisition_date"] = exif['DateTime']
        except Exception:
            pass
            
    return meta

