from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SpatialProfile(BaseModel):
    crs: Optional[str] = None
    bounds: Optional[List[float]] = None  # [minx, miny, maxx, maxy]
    footprint: Optional[Dict] = None  # GeoJSON dict
    width: Optional[int] = None
    height: Optional[int] = None
    gsd_m: Optional[float] = None
    transform: Optional[List[float]] = None


class SensorProfile(BaseModel):
    modality: str
    sensor: str
    bands: Optional[List[str]] = None
    identification_confidence: str = "unknown"
    identification_source: str = "unknown"


class QualityProfile(BaseModel):
    score: float
    source: str = "current_mc1"
    status: str = "unknown"
    metrics: Dict[str, Any] = Field(default_factory=dict)
    reasons: List[str] = Field(default_factory=list)

class ConditionProfile(BaseModel):
    nodata_fraction: float = 0.0
    nodata_status: str = "unknown"
    nodata_source: str = "unknown"
    cloud_fraction: Optional[float] = None
    cloud_status: str = "unknown"
    cloud_source: str = "unknown"


class ObservationProfile(BaseModel):
    observation_id: str
    file_source: str
    file_format: str
    spatial: SpatialProfile
    temporal: Optional[Dict] = None  # {"timestamp": str}
    sensor: SensorProfile
    quality: QualityProfile
    conditions: Optional[ConditionProfile] = None


class CompatibilityProfile(BaseModel):
    hard_compatible: bool = True
    soft_suitability_score: float = 1.0
    hard_failures: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    factors: Dict[str, Any] = Field(default_factory=dict)


class RequestObservationProfile(BaseModel):
    """The aggregate profile for the entire request (1 or 2 images)."""
    observations: List[ObservationProfile]
    spatial_overlap: Optional[float] = None
    coregistration_score: Optional[float] = None
    relationship: str = "unknown"
    task_executable: bool = False
    warnings: List[str] = []
    compatibility: Optional[CompatibilityProfile] = None

    def to_legacy_dict(self, query: str) -> dict:
        """Compatibility adapter for existing MC1 consumers."""
        image_count = len(self.observations)
        
        legacy = {
            "image_count": image_count,
            "query": query,
            "spatial_overlap": self.spatial_overlap,
            "coregistration_score": self.coregistration_score,
            "relationship": self.relationship,
            "quality": {},
            "warnings": self.warnings,
            "task_executable": self.task_executable
        }
        
        footprints = []
        for i, obs in enumerate(self.observations):
            img_key = f"image_{i+1}"
            legacy[img_key] = {
                "modality": obs.sensor.modality,
                "sensor": obs.sensor.sensor,
                "gsd_m": obs.spatial.gsd_m,
                "crs": obs.spatial.crs,
                "filename": obs.file_source  # Always include filename
            }
                
            legacy["quality"][obs.sensor.modality if obs.sensor.modality != "unknown" else img_key] = obs.quality.score
            
            if obs.spatial.footprint:
                legacy["footprint"] = obs.spatial.footprint
                footprints.append(obs.spatial.footprint)
                
            if i == 0 and obs.spatial.transform:
                legacy["affine_transform"] = obs.spatial.transform
                
        if image_count == 1 and not footprints:
            legacy["footprint"] = None

        return legacy
