from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import os
import threading
import time

from agent.schemas import ToolCall, ToolResult

# Specialist modules (transformers, CROMA, ...) are imported lazily inside the adapters so that
# importing the agent layer never loads model code.

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPORT_DIR = os.path.join(_BACKEND_DIR, "exports")


def _elapsed_ms(started_at: float) -> Dict[str, int]:
    return {"duration_ms": int((time.time() - started_at) * 1000)}


def _failed(call: ToolCall, started_at: float, error: str, outputs: Optional[Dict] = None) -> ToolResult:
    return ToolResult(
        call_id=call.call_id,
        tool_id=call.tool_id,
        status="failed",
        outputs=outputs or {},
        evidence_references=[],
        error_information=error,
        execution_metadata=_elapsed_ms(started_at),
    )


def _bound_observation(call: ToolCall, role: str, default: str) -> str:
    binding = call.input_bindings.get(role)
    return binding.observation_id if binding and binding.observation_id else default


class ToolExecutionAdapter(ABC):
    """
    Abstract base class establishing the execution boundary between the
    AgentController and Specialist models.
    """
    @abstractmethod
    def execute(self, call: ToolCall, mc1_profile: Dict, image_tensors: Dict) -> ToolResult:
        pass

    def availability(self) -> Tuple[bool, str]:
        """Cheap check (no model loading) of whether this engine can run in this environment."""
        return True, "available"


# ── PaliGemma (single-image VQA) ─────────────────────────────────────────────

_PALIGEMMA_MODEL_ID = "google/paligemma-3b-pt-224"
_paligemma_lock = threading.Lock()
_paligemma_specialist = None
_paligemma_load_error: Optional[str] = None


def _get_shared_paligemma():
    """Process-wide PaliGemma instance: loading 3B weights per job would exhaust memory."""
    global _paligemma_specialist, _paligemma_load_error
    with _paligemma_lock:
        if _paligemma_load_error:
            raise RuntimeError(_paligemma_load_error)
        if _paligemma_specialist is None:
            from mc4a_vqa.specialist import PaliGemmaVQASpecialist
            try:
                _paligemma_specialist = PaliGemmaVQASpecialist(execution_mode="real")
            except RuntimeError as e:
                _paligemma_load_error = str(e)
                raise
        return _paligemma_specialist


class PaliGemmaExecutionAdapter(ToolExecutionAdapter):
    def __init__(self, execution_mode: str = "real"):
        self.execution_mode = execution_mode
        self._specialist = None
        self._load_error = None

    def availability(self) -> Tuple[bool, str]:
        try:
            from huggingface_hub import try_to_load_from_cache
            index = try_to_load_from_cache(_PALIGEMMA_MODEL_ID, "model.safetensors.index.json")
            if not isinstance(index, str):
                return False, f"MODEL_UNAVAILABLE: {_PALIGEMMA_MODEL_ID} weights are not in the local Hugging Face cache"
            return True, "weights cached"
        except Exception as e:
            return False, f"MODEL_UNAVAILABLE: {e}"

    def _get_specialist(self):
        if self._load_error:
            raise RuntimeError(self._load_error)
        if self._specialist is None:
            try:
                if self.execution_mode == "fixture":
                    from mc4a_vqa.specialist import PaliGemmaVQASpecialist
                    self._specialist = PaliGemmaVQASpecialist(execution_mode="fixture")
                else:
                    self._specialist = _get_shared_paligemma()
            except RuntimeError as e:
                self._load_error = str(e)
                raise e
        return self._specialist

    def execute(self, call: ToolCall, mc1_profile: Dict, image_tensors: Dict) -> ToolResult:
        started_at = time.time()
        try:
            try:
                specialist = self._get_specialist()
            except RuntimeError as e:
                return _failed(call, started_at, str(e))

            import numpy as np
            from PIL import Image
            from mc4a_vqa.evidence_normalizer import to_evidence_object

            obs_id = _bound_observation(call, "primary", "image_1")
            rasters = image_tensors.get("rasters") or {}
            raster = rasters.get(obs_id)

            if raster is not None:
                from raster_io import to_pil_rgb
                from mc4b_temporal.classical import footprint_wgs84
                pil_image = to_pil_rgb(raster)
                footprint = footprint_wgs84(raster)
            else:
                # Tensor inputs (tests / fixture callers without uploaded rasters)
                target_tensor = image_tensors.get("t2" if obs_id == "image_2" else "t1")
                if target_tensor is None:
                    return _failed(call, started_at, "Missing image tensor")
                arr = target_tensor.cpu().detach().numpy()
                if arr.ndim == 4 and arr.shape[0] == 1:
                    arr = arr[0]
                if arr.ndim == 3:
                    if arr.shape[0] in [1, 3]:
                        arr = np.transpose(arr, (1, 2, 0))
                    if arr.shape[2] == 1:
                        arr = arr[:, :, 0]
                if arr.dtype.kind == 'f':
                    arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
                pil_image = Image.fromarray(arr)
                footprint = None

            # The specialist and normaliser read "image_1"; point it at the bound observation.
            target_profile = dict(mc1_profile)
            target_profile["image_1"] = mc1_profile.get(obs_id, mc1_profile.get("image_1", {}))
            target_profile["image_count"] = 1
            target_profile["footprint"] = footprint

            query = call.arguments.get("query", "")
            result = specialist.run(
                image_input=pil_image,
                query=query,
                mc1_profile=target_profile,
                mc2_task_spec=call.arguments
            )

            if "blocked_reason" in result:
                return _failed(call, started_at, f"Quality gate blocked: {result['blocked_reason']}")

            job_id = image_tensors.get("job_id", "job")
            evidence = to_evidence_object(result, target_profile, {"query": query, **call.arguments}, f"{job_id}_{call.call_id}")
            for ev in evidence:
                ev["source_input"] = target_profile["image_1"].get("filename", obs_id)
                if self.execution_mode == "fixture":
                    ev["source_model"] = "PALIGEMMA_VQA_TOOL (FIXTURE)"

            clean_outputs = {k: v for k, v in result.items() if k not in ["spatial_evidence", "raw_tensors"]}
            clean_outputs["evidence"] = evidence
            clean_outputs["observation_id"] = obs_id

            return ToolResult(
                call_id=call.call_id,
                tool_id=call.tool_id,
                status="succeeded",
                outputs=clean_outputs,
                evidence_references=[e["evidence_id"] for e in evidence],
                execution_metadata=_elapsed_ms(started_at)
            )

        except Exception as e:
            return _failed(call, started_at, str(e))


# ── ChangeMamba (learned temporal model, CUDA only) ─────────────────────────

class ChangeMambaExecutionAdapter(ToolExecutionAdapter):
    def __init__(self, execution_mode: str = "real"):
        self.execution_mode = execution_mode
        self._specialist = None

    def availability(self) -> Tuple[bool, str]:
        from mc4b_temporal.backbone import get_backbone
        try:
            get_backbone("changemamba")
            return True, "available"
        except NotImplementedError as e:
            return False, f"MODEL_UNAVAILABLE: {e}"
        except Exception as e:
            return False, f"MODEL_UNAVAILABLE: {e}"

    def _get_specialist(self):
        if self._specialist is None:
            from mc4b_temporal.tool_adapter import ChangeMambaAdapter
            self._specialist = ChangeMambaAdapter()
        return self._specialist

    def execute(self, call: ToolCall, mc1_profile: Dict, image_tensors: Dict) -> ToolResult:
        started_at = time.time()
        try:
            t1 = image_tensors.get("t1")
            t2 = image_tensors.get("t2")

            if t1 is None or t2 is None:
                return _failed(call, started_at, "Missing image tensors")

            if self.execution_mode == "fixture":
                return self._execute_fixture_fallback(call, mc1_profile, t1, t2, call.arguments.get("query", ""), started_at)

            specialist = self._get_specialist()
            result = specialist.execute(
                mc1_profile,
                t1,
                t2,
                query=call.arguments.get("query", "")
            )

            if result.get("status") == "PRECONDITION_FAILED":
                return _failed(call, started_at, f"Precondition failed: {result.get('failed')}")

            # MODEL_UNAVAILABLE is a distinct, non-fabricated failure
            if result.get("status") != "SUCCESS":
                return _failed(
                    call, started_at,
                    f"MODEL_UNAVAILABLE: {result.get('reason', 'ChangeMamba model cannot be loaded')}",
                    outputs={"status": result.get("status", "MODEL_UNAVAILABLE")},
                )

            clean_outputs = {k: v for k, v in result.items() if k not in ["change_map", "binary_change_mask", "raw_tensors"]}
            return ToolResult(
                call_id=call.call_id,
                tool_id=call.tool_id,
                status="succeeded",
                outputs=clean_outputs,
                evidence_references=[],
                execution_metadata=_elapsed_ms(started_at)
            )

        except Exception as e:
            return _failed(call, started_at, str(e))

    def _execute_fixture_fallback(self, call, mc1_profile, t1, t2, query, started_at):
        if call.arguments.get("force_fail") is True or "fail" in query.lower():
            return _failed(call, started_at, "Simulated failure via query for D3 scenario.")
        evidence = [{
            "evidence_id": f"fixture_{call.call_id}_change",
            "claim": "[FIXTURE] Synthetic change result; no change model was executed.",
            "evidence_type": "bitemporal_change_summary",
            "spatial_region": None,
            "modality": mc1_profile.get("image_1", {}).get("modality", "optical"),
            "modality_contribution": {},
            "timestamp": {"before": "image_1", "after": "image_2"},
            "source_model": "CHANGE_MAMBA_TOOL (FIXTURE)",
            "source_input": {"before": "image_1", "after": "image_2"},
            "confidence": 0.5,
            "processing_parameters": {"fixture_mode": True},
        }]
        return ToolResult(
            call_id=call.call_id,
            tool_id=call.tool_id,
            status="succeeded",
            outputs={"change_detected": True, "fixture_mode": True, "evidence": evidence},
            evidence_references=[e["evidence_id"] for e in evidence],
            execution_metadata=_elapsed_ms(started_at)
        )


# ── Classical change detection (runs anywhere, on the uploaded pixels) ──────

class ClassicalChangeDetectionAdapter(ToolExecutionAdapter):
    MIN_OVERLAP = 0.6

    def __init__(self, execution_mode: str = "real"):
        self.execution_mode = execution_mode

    def execute(self, call: ToolCall, mc1_profile: Dict, image_tensors: Dict) -> ToolResult:
        started_at = time.time()
        try:
            from mc4b_temporal.classical import detect_change, render_change_overlay, footprint_wgs84, image_bounds_wgs84

            rasters = image_tensors.get("rasters") or {}
            before_id = _bound_observation(call, "before", "image_1")
            after_id = _bound_observation(call, "after", "image_2")
            before, after = rasters.get(before_id), rasters.get(after_id)
            if before is None or after is None:
                return _failed(call, started_at, "INPUT_INSUFFICIENT: classical change detection needs two loaded observations")

            mod_before = mc1_profile.get(before_id, {}).get("modality", "unknown")
            mod_after = mc1_profile.get(after_id, {}).get("modality", "unknown")
            if mod_before != mod_after:
                return _failed(call, started_at, f"INPUT_INVALID: both observations must share a modality (got {mod_before} and {mod_after})")
            modality = mod_before
            if modality not in ("optical", "sar"):
                # Unknown modality: decide from the pixels rather than guessing a sensor
                modality = "sar" if before.data.shape[0] <= 2 else "optical"

            if before.is_georeferenced and after.is_georeferenced:
                if before.crs != after.crs:
                    return _failed(call, started_at, f"INPUT_INVALID: CRS differs ({before.crs} vs {after.crs}); reprojection is not implemented")
                overlap = mc1_profile.get("spatial_overlap")
                if overlap is None or overlap < self.MIN_OVERLAP:
                    return _failed(call, started_at, f"INPUT_INVALID: spatial overlap {overlap} is below {self.MIN_OVERLAP}")
            elif before.is_georeferenced != after.is_georeferenced:
                return _failed(call, started_at, "INPUT_INVALID: only one observation is georeferenced; cannot align the pair")

            detection = detect_change(before, after, modality)
            stats = detection["statistics"]

            job_id = image_tensors.get("job_id", "job")
            os.makedirs(EXPORT_DIR, exist_ok=True)
            overlay_path = render_change_overlay(detection, os.path.join(EXPORT_DIR, f"export_{job_id}_heatmap.png"))

            method = detection["method"]
            source_model = f"CLASSICAL_CHANGE_DETECTION ({method})"
            before_name = mc1_profile.get(before_id, {}).get("filename", before_id)
            after_name = mc1_profile.get(after_id, {}).get("filename", after_id)
            area_txt = f", about {stats['changed_area_m2']:,.0f} m²" if stats["changed_area_m2"] is not None else ""
            common_params = {
                "method": method,
                "threshold": stats["threshold"],
                "threshold_unit": stats["threshold_unit"],
                "otsu_threshold": stats["otsu_threshold"],
                "semantic_classes": "not_available (classical detector localises change only)",
            }

            evidence = []
            if stats["region_count"] == 0:
                summary_claim = (
                    f"No significant change detected between {before_name} (before) and {after_name} (after) "
                    f"at a threshold of {stats['threshold']} {stats['threshold_unit']}."
                )
            else:
                summary_claim = (
                    f"Change detected between {before_name} (before) and {after_name} (after): "
                    f"{stats['region_count']} region(s) covering {stats['changed_pixel_pct']:.2f}% of the scene{area_txt}."
                )
            evidence.append({
                "evidence_id": f"{job_id}_{call.call_id}_summary",
                "claim": summary_claim,
                "evidence_type": "bitemporal_change_summary",
                "spatial_region": footprint_wgs84(before),
                "modality": modality,
                "modality_contribution": {modality: 1.0},
                "timestamp": {"before": before_id, "after": after_id},
                "source_model": source_model,
                "source_input": {"before": before_name, "after": after_name},
                "confidence": round(detection["separability"], 3),
                "processing_parameters": {
                    **common_params,
                    "confidence_source": "Otsu class separability of the change magnitude (uncalibrated)",
                    "spatial_region_is_full_image": True,
                },
            })

            for region in detection["regions"][:5]:
                strength = min(1.0, region["mean_magnitude"] / (2 * max(detection["threshold"], 1e-6)))
                size_txt = f"{region['area_m2']:,.0f} m²" if region["area_m2"] is not None else f"{region['area_pixels']} px"
                evidence.append({
                    "evidence_id": f"{job_id}_{call.call_id}_region_{region['region_index']}",
                    "claim": (
                        f"Change region {region['region_index'] + 1}: {size_txt}, mean "
                        f"{'increase' if region['direction'] == 'increase' else 'decrease'} of "
                        f"{abs(region['mean_change']):.2f} {stats['threshold_unit']}."
                    ),
                    "evidence_type": "bitemporal_change_region",
                    "spatial_region": region["geometry"],
                    "modality": modality,
                    "modality_contribution": {modality: 1.0},
                    "timestamp": {"before": before_id, "after": after_id},
                    "source_model": source_model,
                    "source_input": {"before": before_name, "after": after_name},
                    "confidence": round(strength, 3),
                    "processing_parameters": {
                        **common_params,
                        "bbox_pixel": region["bbox_pixel"],
                        "confidence_source": "change strength = mean magnitude / (2 x threshold), uncalibrated",
                    },
                })

            outputs = {
                "method": method,
                "modality": modality,
                "change_detected": stats["region_count"] > 0,
                "change_statistics": stats,
                "regions": [{k: v for k, v in r.items()} for r in detection["regions"]],
                "overlay_png": overlay_path,
                "overlay_bounds": image_bounds_wgs84(before),
                "before_observation": before_id,
                "after_observation": after_id,
                "notes": detection["notes"],
                "evidence": evidence,
            }
            return ToolResult(
                call_id=call.call_id,
                tool_id=call.tool_id,
                status="succeeded",
                outputs=outputs,
                evidence_references=[e["evidence_id"] for e in evidence],
                execution_metadata=_elapsed_ms(started_at),
            )
        except Exception as e:
            return _failed(call, started_at, f"TOOL_EXECUTION_FAILED: {e}")


class MockToolAdapter(ToolExecutionAdapter):
    """Deterministic fake specialist for testing controller behavior."""
    def __init__(self, should_fail: bool = False, fail_reason: str = "Mock failure"):
        self.should_fail = should_fail
        self.fail_reason = fail_reason

    def execute(self, call: ToolCall, mc1_profile: Dict, image_tensors: Dict) -> ToolResult:
        if self.should_fail:
            return ToolResult(
                call_id=call.call_id,
                tool_id=call.tool_id,
                status="failed",
                outputs={},
                evidence_references=[],
                error_information=self.fail_reason,
                execution_metadata={"duration_ms": 10}
            )

        # Optional testing hook to fail based on arguments
        if call.arguments.get("force_fail") is True:
            return ToolResult(
                call_id=call.call_id,
                tool_id=call.tool_id,
                status="failed",
                outputs={},
                evidence_references=[],
                error_information="Forced failure via arguments",
                execution_metadata={"duration_ms": 10}
            )

        return ToolResult(
            call_id=call.call_id,
            tool_id=call.tool_id,
            status="succeeded",
            outputs={"mock_output": True, "processed_args": call.arguments},
            evidence_references=["ev_mock_1"],
            execution_metadata={"duration_ms": 15}
        )


# ── CROMA optical + SAR fusion ───────────────────────────────────────────────

CROMA_WEIGHTS = os.path.join(_BACKEND_DIR, "mc4c", "weights", "CROMA_base.pt")
SEMANTIC_WEIGHTS = os.path.join(_BACKEND_DIR, "mc4c", "weights", "model.safetensors")
_croma_lock = threading.Lock()
_croma_engine = None


def _get_shared_croma():
    global _croma_engine
    with _croma_lock:
        if _croma_engine is None:
            from mc4c.engine import MC4CEngine
            _croma_engine = MC4CEngine(croma_weights_path=CROMA_WEIGHTS, semantic_weights_path=SEMANTIC_WEIGHTS)
        return _croma_engine


class CromaExecutionAdapter(ToolExecutionAdapter):
    """
    Adapter for the Phase 4 CROMA Cross-Modal Specialist.
    Expects exactly one multispectral optical (Sentinel-2) and one dual-pol SAR (Sentinel-1) observation.
    """
    MIN_OPTICAL_BANDS = 10
    SAR_BANDS = 2

    def __init__(self, execution_mode: str = "real"):
        self.execution_mode = execution_mode

    def availability(self) -> Tuple[bool, str]:
        if not os.path.exists(CROMA_WEIGHTS):
            return False, f"MODEL_UNAVAILABLE: CROMA weights not found at {CROMA_WEIGHTS}"
        if os.getenv("SATQUERY_ENABLE_UNVALIDATED_CROMA") != "1":
            return False, (
                "UNVALIDATED: CROMA runs, but its optical/SAR agreement score does not distinguish matching from "
                "non-matching BigEarthNet pairs (40-pair check: matched 0.011 vs unmatched 0.011 cosine), so it is "
                "withheld from answers. Set SATQUERY_ENABLE_UNVALIDATED_CROMA=1 to enable for research."
            )
        return True, "weights present (unvalidated outputs enabled by environment flag)"

    def execute(self, call: ToolCall, mc1_profile: Dict, image_tensors: Dict) -> ToolResult:
        started_at = time.time()
        try:
            optical_id = sar_id = None
            if call.input_bindings.get("optical") and call.input_bindings.get("sar"):
                optical_id = call.input_bindings["optical"].observation_id
                sar_id = call.input_bindings["sar"].observation_id
            else:
                for key in ["image_1", "image_2"]:
                    modality = (mc1_profile.get(key) or {}).get("modality", "").lower()
                    if modality == "optical" and not optical_id:
                        optical_id = key
                    elif modality == "sar" and not sar_id:
                        sar_id = key

            if not optical_id or not sar_id:
                return _failed(call, started_at, "Missing optical or SAR observation in mc1_profile")

            if self.execution_mode == "fixture":
                return self._execute_fixture_fallback(call, mc1_profile, call.arguments.get("query", ""), optical_id, sar_id, started_at)

            if not os.path.exists(CROMA_WEIGHTS):
                return _failed(call, started_at, f"MODEL_UNAVAILABLE: CROMA weights not found at {CROMA_WEIGHTS}", outputs={"status": "MODEL_UNAVAILABLE"})

            rasters = image_tensors.get("rasters") or {}
            optical, sar = rasters.get(optical_id), rasters.get(sar_id)
            if optical is None or sar is None:
                return _failed(call, started_at, "INPUT_INSUFFICIENT: optical/SAR rasters were not loaded; no fused output is produced")
            if optical.data.shape[0] < self.MIN_OPTICAL_BANDS or sar.data.shape[0] != self.SAR_BANDS:
                return _failed(
                    call, started_at,
                    f"INPUT_INVALID: CROMA needs a >= {self.MIN_OPTICAL_BANDS}-band Sentinel-2 image and a "
                    f"{self.SAR_BANDS}-band (VV/VH) Sentinel-1 image; got {optical.data.shape[0]} and {sar.data.shape[0]} bands"
                )

            query = call.arguments.get("query", "")
            engine = _get_shared_croma()
            result_dict = engine.process({
                "optical_image": optical.path,
                "sar_image": sar.path,
                "query": query,
                "input_profile": {"optical": mc1_profile.get(optical_id, {}), "sar": mc1_profile.get(sar_id, {})},
                "task_spec": dict(call.arguments),
            })

            from mc4b_temporal.classical import footprint_wgs84
            job_id = image_tensors.get("job_id", "job")
            status = result_dict["verification_status"]
            score = float(result_dict["confidence_score"])
            cosine = 2 * score - 1
            agreement = {
                "MATCH": "strongly agree, consistent with the two images showing the same scene",
                "PARTIAL": "partially agree",
                "MISMATCH": "disagree, so the images may not show the same scene",
            }.get(status, status)
            evidence = [{
                "evidence_id": f"{job_id}_{call.call_id}_croma_consistency",
                "claim": f"[Unvalidated research output] CROMA optical and SAR embeddings {agreement} (cosine similarity {cosine:.2f}, {status}).",
                "evidence_type": "cross_modal_consistency_unvalidated",
                "spatial_region": footprint_wgs84(optical),
                "modality": "optical+sar",
                "modality_contribution": {"optical": 0.5, "sar": 0.5},
                "timestamp": {"optical": optical_id, "sar": sar_id},
                "source_model": "CROMA_BASE (optical+SAR encoders)",
                "source_input": {"optical": optical.filename, "sar": sar.filename},
                "confidence": round(score, 3),
                "processing_parameters": {
                    "fusion_method": result_dict.get("fusion_method"),
                    "confidence_source": "(cosine similarity + 1) / 2 of CROMA global embeddings (uncalibrated)",
                    "spatial_region_is_full_image": True,
                },
            }]

            clean_outputs = {
                "fusion_method": result_dict.get("fusion_method"),
                "verification_status": status,
                "embedding_cosine_similarity": round(cosine, 4),
                "land_cover_tags_unvalidated": result_dict.get("semantic_metadata", {}).get("joint_tags", []),
                "land_cover_tags_note": (
                    "Not used as evidence: the BigEarthNet head's input conventions are unverified "
                    "(measured micro-F1 0.28 on 80 BigEarthNet test patches)."
                ),
                "evidence": evidence,
            }
            return ToolResult(
                call_id=call.call_id,
                tool_id=call.tool_id,
                status="succeeded",
                outputs=clean_outputs,
                evidence_references=[e["evidence_id"] for e in evidence],
                execution_metadata=_elapsed_ms(started_at)
            )

        except Exception as e:
            return _failed(call, started_at, str(e))

    def _execute_fixture_fallback(self, call, mc1_profile, query, optical_id, sar_id, started_at):
        """Fixture-mode only: deterministic zero tensors, explicitly flagged as fixture output."""
        if call.arguments.get("force_fail") is True or "fail" in query.lower():
            return _failed(call, started_at, "Simulated failure via query for D3 scenario.")
        import torch
        from mc4c.cross_modal_evidence import CrossModalEvidenceAdapter
        from mc4c.fusion_schema import FusedTokenRepresentation
        from mc4c.token_schema import TokenSpatialIdentity
        from mc4c.query_schema import QueryRepresentation

        num_tokens = 225
        opt_feat = torch.zeros(1, num_tokens, 768)
        sar_feat = torch.zeros(1, num_tokens, 768)

        spatial_identities = [
            TokenSpatialIdentity(
                original_index=i,
                row=i // 15,
                column=i % 15,
                bounds=[0, 0, 1, 1]
            ) for i in range(num_tokens)
        ]

        fused_rep = FusedTokenRepresentation(
            grid_height=15,
            grid_width=15,
            batch_size=1,
            num_tokens=num_tokens,
            spatial_identities=spatial_identities,
            fused_tokens=torch.zeros(1, num_tokens, 768),
            query_context=QueryRepresentation(
                original_query=query,
                primary_task="cross_modal_fusion",
                embedding=torch.zeros(1, 384)
            )
        )

        ev_adapter = CrossModalEvidenceAdapter()
        candidates = ev_adapter.generate_candidates(
            fused_rep,
            optical_obs={"observation_id": optical_id},
            sar_obs={"observation_id": sar_id},
            mc1_compatibility={"status": "compatible"},
            indices=[0]
        )

        mc5_evidence = [c.to_mc5_evidence() for c in candidates]
        for ev in mc5_evidence:
            ev["source_model"] = f"{ev.get('source_model', 'CROMA')} (FIXTURE)"

        clean_outputs = {
            "optical_tokens_shape": list(opt_feat.shape),
            "sar_tokens_shape": list(sar_feat.shape),
            "joint_tokens_shape": [1, 225, 768],
            "fusion_method": "CROMA_query_conditioned",
            "evidence": mc5_evidence,
            "fixture_mode": True
        }

        return ToolResult(
            call_id=call.call_id,
            tool_id=call.tool_id,
            status="succeeded",
            outputs=clean_outputs,
            evidence_references=[e["evidence_id"] for e in mc5_evidence],
            execution_metadata=_elapsed_ms(started_at)
        )


def build_adapters(execution_mode: str = "real") -> Dict[str, ToolExecutionAdapter]:
    return {
        "single_image_vqa": PaliGemmaExecutionAdapter(execution_mode=execution_mode),
        "temporal_change_analysis": ChangeMambaExecutionAdapter(execution_mode=execution_mode),
        "classical_change_detection": ClassicalChangeDetectionAdapter(execution_mode=execution_mode),
        "optical_sar_fusion": CromaExecutionAdapter(execution_mode=execution_mode),
    }
