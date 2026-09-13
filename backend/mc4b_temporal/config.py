"""
config.py — MC4B configuration: thresholds, feature flags, backbone selection.
"""

# ── Backbone Selection ────────────────────────────────────────────
# Set to "changemamba" when CUDA hardware + mamba-ssm is available.
# Default: "lightweight_cnn" (CPU-compatible stand-in)
BACKBONE = "lightweight_cnn"

# ── Feature Flags ─────────────────────────────────────────────────
ENABLE_CAPTIONER = False          # DeltaVLM-style change VQA/caption head
ENABLE_SEMANTIC_CHANGE = True     # Per-class semantic change detection

# ── Precondition Thresholds ───────────────────────────────────────
MIN_SPATIAL_OVERLAP = 0.6         # Minimum IoU between T1 and T2 footprints
MIN_COREGISTRATION_SCORE = 0.5    # Minimum co-registration quality
GSD_TOLERANCE_FACTOR = 2.0        # Max allowed GSD ratio between T1 and T2

# ── Model Parameters ─────────────────────────────────────────────
INPUT_SIZE = 256                  # Input tile size (H=W)
NUM_CLASSES = 5                   # Semantic change classes (incl. no-change)
CONFIDENCE_CALIBRATION = True     # Apply temperature scaling to confidence

# ── Semantic Change Class Labels ──────────────────────────────────
CHANGE_CLASSES = [
    "no_change",
    "built_up_gain",
    "built_up_loss",
    "vegetation_loss",
    "vegetation_gain",
]

# ── Verification / Re-plan ────────────────────────────────────────
LOW_CONFIDENCE_THRESHOLD = 0.1    # Below this → RE-PLAN REQUIRED
MAX_REPLAN_ATTEMPTS = 2           # Prevent infinite re-plan loops

# ── Computational Cost ────────────────────────────────────────────
COMPUTATIONAL_COST_LABEL = "medium"
COMPUTATIONAL_COST_FLOPS_ESTIMATE = 12e9  # ~12 GFLOPs for 256×256 pair
