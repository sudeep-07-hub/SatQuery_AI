"""
test_smoke.py — Phase 1 smoke test: all modules import without error.
"""

def test_import_mc4b_config():
    from mc4b_temporal import config
    assert hasattr(config, "BACKBONE")

def test_import_mc4b_backbone():
    from mc4b_temporal.backbone import BackboneBase, LightweightCNNBackbone, get_backbone
    assert callable(get_backbone)

def test_import_mc4b_change_head():
    from mc4b_temporal.change_head import BinaryChangeHead, SemanticChangeHead, ChangeDetector

def test_import_mc4b_semantics():
    from mc4b_temporal.semantics import extract_change_types

def test_import_mc4b_captioner():
    from mc4b_temporal.captioner import generate_change_caption

def test_import_mc4b_localization():
    from mc4b_temporal.localization import mask_to_regions, pixel_to_geo, regions_to_geojson, compute_change_statistics

def test_import_mc4b_tool_adapter():
    from mc4b_temporal.tool_adapter import ChangeMambaAdapter, CHANGE_MAMBA_TOOL, validate_preconditions

def test_import_mc4b_evidence_normalizer():
    from mc4b_temporal.evidence_normalizer import normalize_to_evidence

def test_import_mc3_planner():
    from mc3_planner.tool_registry import ToolRegistry
    from mc3_planner.workflow_planner import build_workflow_plan
    from mc3_planner.dispatcher import execute_plan

def test_import_mc5_evidence():
    from mc5_evidence.schemas import validate_evidence_object
    from mc5_evidence.evidence_graph import EvidenceGraph

def test_import_mc6_verification():
    from mc6_verification.verifier import Verifier
    from mc6_verification.conflict_detector import detect_conflicts

def test_backbone_instantiation():
    """Smoke-test: create a backbone and run a forward pass."""
    import torch
    from mc4b_temporal.backbone import get_backbone

    backbone = get_backbone("lightweight_cnn", in_channels=3)
    t1 = torch.rand(1, 3, 64, 64)
    t2 = torch.rand(1, 3, 64, 64)
    feats1, feats2 = backbone.encode_pair(t1, t2)
    assert len(feats1) == 3
    assert len(feats2) == 3
    # Check shapes decrease at each scale
    assert feats1[0].shape[2] > feats1[1].shape[2] > feats1[2].shape[2]

def test_tool_registry_entry_schema():
    """Smoke-test: CHANGE_MAMBA_TOOL has all required MC3.1 fields."""
    from mc4b_temporal.tool_adapter import CHANGE_MAMBA_TOOL
    from mc3_planner.tool_registry import REQUIRED_TOOL_FIELDS
    missing = REQUIRED_TOOL_FIELDS - set(CHANGE_MAMBA_TOOL.keys())
    assert not missing, f"Missing fields: {missing}"
