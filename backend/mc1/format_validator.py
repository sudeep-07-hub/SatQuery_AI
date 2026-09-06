import os

ALLOWED_EXTENSIONS = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}

# Magic numbers mapping (bytes)
MAGIC_NUMBERS = {
    "png": [b'\x89\x50\x4E\x47\x0D\x0A\x1A\x0A'],
    "jpeg": [b'\xFF\xD8\xFF'],
    "tiff": [b'\x49\x49\x2A\x00', b'\x4D\x4D\x00\x2A', b'\x49\x49\x2B\x00', b'\x4D\x4D\x00\x2B']
}

def validate_format(filename: str, file_bytes: bytes) -> tuple[bool, str, str]:
    """
    Validates file extension and magic number.
    Returns (is_valid, error_reason, detected_format).
    """
    ext = os.path.splitext(filename)[1].lower()
    
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported extension: {ext}", "unknown"
        
    # Sniff magic number
    detected_format = "unknown"
    for fmt, signatures in MAGIC_NUMBERS.items():
        for sig in signatures:
            if file_bytes.startswith(sig):
                detected_format = fmt
                break
        if detected_format != "unknown":
            break
            
    if detected_format == "unknown":
        return False, "File signature does not match accepted formats (PNG, JPEG, TIFF)", "unknown"
        
    # Match extension with detected format roughly
    if detected_format == "png" and ext not in ['.png']:
        return False, f"Extension {ext} does not match detected format PNG", detected_format
    if detected_format == "jpeg" and ext not in ['.jpg', '.jpeg']:
        return False, f"Extension {ext} does not match detected format JPEG", detected_format
    if detected_format == "tiff" and ext not in ['.tif', '.tiff']:
        return False, f"Extension {ext} does not match detected format TIFF", detected_format
        
    return True, "", detected_format
