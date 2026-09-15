import pytest
import asyncio
from unittest.mock import patch, MagicMock
from mc1.pipeline import run_mc1_pipeline
from mc1.schemas import RequestObservationProfile

class MockUploadFile:
    def __init__(self, filename="test.tif"):
        self.filename = filename
    async def read(self):
        return b"mock_data"
    async def seek(self, offset):
        pass

@pytest.fixture
def mock_pipeline_deps():
    with patch("mc1.pipeline.validate_format") as mock_validate, \
         patch("mc1.pipeline.extract_metadata") as mock_extract:
        
        mock_validate.return_value = (True, "", "tiff")
        
        # Default mock meta
        def default_extract(*args, **kwargs):
            return {
                "crs": "EPSG:32633",
                "bounds": MagicMock(left=0, bottom=0, right=10, top=10),
                "width": 100,
                "height": 100,
                "transform": [1.0, 0.0, 0.0, 0.0, -1.0, 10.0],
                "image_statistics": {
                    "valid_fraction": 1.0,
                    "std": 50.0,
                    "mean": 100.0
                },
                "tags": {"SENSOR": "Sentinel-2"}
            }
        mock_extract.side_effect = default_extract
        
        yield mock_validate, mock_extract

@pytest.mark.asyncio
async def test_integration_single_optical(mock_pipeline_deps):
    mock_validate, mock_extract = mock_pipeline_deps
    
    files = [MockUploadFile("optical.tif")]
    result_dict = await run_mc1_pipeline(files, "Find cars")
    
    # Verify legacy output is functional
    assert result_dict["image_count"] == 1
    assert result_dict["task_executable"] is True
    assert result_dict["image_1"]["modality"] == "optical"
    
@pytest.mark.asyncio
async def test_integration_optical_sar_pair(mock_pipeline_deps):
    mock_validate, mock_extract = mock_pipeline_deps
    
    def side_effect_extract(file_bytes, fmt):
        base = {
            "crs": "EPSG:32633",
            "bounds": MagicMock(left=0, bottom=0, right=10, top=10),
            "width": 100,
            "height": 100,
            "transform": [1.0, 0.0, 0.0, 0.0, -1.0, 10.0],
            "image_statistics": {
                "valid_fraction": 1.0,
                "std": 50.0,
                "mean": 100.0
            }
        }
        if b"opt" in file_bytes: # just a hack for mocking since we don't vary bytes
            pass
        return base
        
    # We will just patch the classifier directly to force optical and sar since our bytes are the same
    with patch("mc1.pipeline.EvidenceBasedClassifier.classify") as mock_classify:
        mock_classify.side_effect = [
            ("optical", "Sentinel-2", "high", "metadata"),
            ("sar", "Sentinel-1", "high", "metadata")
        ]
        
        files = [MockUploadFile("opt.tif"), MockUploadFile("sar.tif")]
        result_dict = await run_mc1_pipeline(files, "Compare")
        
        assert result_dict["image_count"] == 2
        assert result_dict["task_executable"] is True # Hard compatible!
        assert result_dict["spatial_overlap"] == 1.0
        assert result_dict["image_1"]["modality"] == "optical"
        assert result_dict["image_2"]["modality"] == "sar"

@pytest.mark.asyncio
async def test_integration_zero_overlap(mock_pipeline_deps):
    mock_validate, mock_extract = mock_pipeline_deps
    
    def side_effect_extract(*args):
        # We will toggle bounds based on a static counter to mock two different locations
        if not hasattr(side_effect_extract, "call_count"):
            side_effect_extract.call_count = 0
            
        if side_effect_extract.call_count == 0:
            bounds = MagicMock(left=0, bottom=0, right=10, top=10)
        else:
            bounds = MagicMock(left=20, bottom=20, right=30, top=30)
            
        side_effect_extract.call_count += 1
        
        return {
            "crs": "EPSG:32633",
            "bounds": bounds,
            "image_statistics": {"valid_fraction": 1.0, "std": 50.0},
            "tags": {"SENSOR": "Sentinel-2"}
        }
        
    mock_extract.side_effect = side_effect_extract
    
    files = [MockUploadFile("1.tif"), MockUploadFile("2.tif")]
    result_dict = await run_mc1_pipeline(files, "Compare")
    
    assert result_dict["image_count"] == 2
    assert result_dict["task_executable"] is False # Zero overlap
    assert result_dict["spatial_overlap"] == 0.0
    assert any("No spatial overlap" in w for w in result_dict["warnings"])

@pytest.mark.asyncio
async def test_integration_invalid_input(mock_pipeline_deps):
    mock_validate, mock_extract = mock_pipeline_deps
    mock_validate.return_value = (False, "Bad format", "unknown")
    
    files = [MockUploadFile("bad.png")]
    result_dict = await run_mc1_pipeline(files, "Look")
    
    assert result_dict["task_executable"] is False
    assert result_dict["image_1"]["modality"] == "unknown"
    assert any("unknown or invalid" in w for w in result_dict["warnings"])
