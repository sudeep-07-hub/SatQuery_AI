"""
mc4b_temporal — ChangeMamba Bi-Temporal Change Detection Engine (MC4B)

Provides the temporal specialist for SATQUERY AI's agentic pipeline.
Processes bi-temporal image pairs (T1, T2) and produces change maps,
confidence scores, georeferenced change regions, and Evidence Objects.

Architecture note:
    The backbone is swappable via config.BACKBONE_CLASS. The default
    LightweightCNNBackbone is a CPU-compatible stand-in for the real
    ChangeMamba (Mamba-SSM) backbone, which requires CUDA hardware.
    All agentic contracts (MC3 Tool registry, MC5 Evidence, MC6 triggers)
    are backbone-agnostic and fully functional regardless of which
    backbone is active.
"""
