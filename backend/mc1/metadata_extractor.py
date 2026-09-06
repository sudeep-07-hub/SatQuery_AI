import io
import rasterio
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
        "gsd_m": None
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
                    
                    # Try to extract date from tags
                    t = dataset.tags()
                    if 'TIFFTAG_DATETIME' in t:
                        meta["acquisition_date"] = t['TIFFTAG_DATETIME']
                        
                    # Calculate GSD (assume transform scale gives pixel size)
                    if dataset.transform:
                        meta["gsd_m"] = abs(dataset.transform.a) # Pixel width
        except Exception as e:
            # Add warning later
            pass
            
    elif detected_format in ["png", "jpeg"]:
        try:
            img = Image.open(io.BytesIO(file_bytes))
            meta["width"] = img.width
            meta["height"] = img.height
            meta["band_count"] = len(img.getbands())
            
            # Try EXIF
            if hasattr(img, '_getexif') and img._getexif():
                exif = {TAGS.get(k, k): v for k, v in img._getexif().items()}
                if 'DateTimeOriginal' in exif:
                    meta["acquisition_date"] = exif['DateTimeOriginal']
                elif 'DateTime' in exif:
                    meta["acquisition_date"] = exif['DateTime']
        except Exception:
            pass
            
    return meta
