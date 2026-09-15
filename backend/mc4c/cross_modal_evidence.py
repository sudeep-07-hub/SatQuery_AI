import hashlib
import time
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from .fusion_schema import FusedTokenRepresentation
from .token_schema import TokenSpatialIdentity

class CrossModalEvidenceCandidate(BaseModel):
    """
    Typed evidence candidate representing cross-modal spatial information
    derived from query-conditioned fusion.
    """
    evidence_id: str
    query_context: str
    optical_observation_id: str
    sar_observation_id: str
    token_identity: TokenSpatialIdentity
    representation_provenance: str
    evidence_type: str = "cross_modal_spatial"
    modality: str = "optical+sar"
    
    def to_mc5_evidence(self) -> Dict[str, Any]:
        """
        Converts the candidate into the authoritative MC5 Evidence Object format.
        """
        return {
            "evidence_id": self.evidence_id,
            "claim": "Unverified spatial multimodal candidate",
            "evidence_type": self.evidence_type,
            "spatial_region": {
                "type": "Polygon",
                "coordinates": [[
                    [self.token_identity.bounds[1], self.token_identity.bounds[0]], # x, y
                    [self.token_identity.bounds[3], self.token_identity.bounds[0]],
                    [self.token_identity.bounds[3], self.token_identity.bounds[2]],
                    [self.token_identity.bounds[1], self.token_identity.bounds[2]],
                    [self.token_identity.bounds[1], self.token_identity.bounds[0]],
                ]],
                "bounds": self.token_identity.bounds
            },
            "modality": self.modality,
            "modality_contribution": {"optical": 0.5, "sar": 0.5},
            "timestamp": time.time(), # Mocked timestamp for schema compliance
            "source_model": self.representation_provenance,
            "source_input": {
                "optical_id": self.optical_observation_id,
                "sar_id": self.sar_observation_id,
                "token_index": self.token_identity.original_index
            },
            "confidence": 0.0,
            "processing_parameters": {
                "query_context": self.query_context
            }
        }


class CrossModalEvidenceAdapter:
    """
    Deterministic interface that generates cross-modal evidence candidates
    from a FusedTokenRepresentation.
    """
    
    @staticmethod
    def _generate_evidence_id(
        query_context: str, 
        opt_id: str, 
        sar_id: str, 
        token_index: int
    ) -> str:
        """Deterministically generates an evidence ID based on provenance invariants."""
        raw = f"{query_context}_{opt_id}_{sar_id}_{token_index}"
        hash_val = hashlib.md5(raw.encode('utf-8')).hexdigest()[:12]
        return f"ev_xmodal_{hash_val}"
        
    def generate_candidates(
        self,
        fused_rep: FusedTokenRepresentation,
        optical_obs: Dict[str, Any],
        sar_obs: Dict[str, Any],
        mc1_compatibility: Dict[str, Any],
        indices: Optional[List[int]] = None
    ) -> List[CrossModalEvidenceCandidate]:
        """
        Generates deterministic evidence candidates for the specified spatial tokens.
        """
        # Validate inputs
        if not optical_obs or "observation_id" not in optical_obs:
            raise ValueError("Missing or invalid optical observation")
            
        if not sar_obs or "observation_id" not in sar_obs:
            raise ValueError("Missing or invalid SAR observation")
            
        # Check MC1 Compatibility
        status = mc1_compatibility.get("status", "unknown")
        if status == "hard_incompatible":
            raise ValueError("Refusing candidate generation: MC1 hard incompatibility detected.")
            
        opt_id = optical_obs["observation_id"]
        sar_id = sar_obs["observation_id"]
        query_ctx = fused_rep.query_context.original_query
        
        # Determine subset of indices to generate
        total_tokens = fused_rep.num_tokens
        if indices is None:
            # We don't automatically explode the evidence graph unless requested
            # Bounded generation fallback to token 0
            indices = [0]
            
        candidates = []
        
        for idx in indices:
            if not (0 <= idx < total_tokens):
                raise IndexError(f"Token index {idx} out of bounds.")
                
            spatial_id = fused_rep.spatial_identities[idx]
            
            ev_id = self._generate_evidence_id(query_ctx, opt_id, sar_id, spatial_id.original_index)
            
            candidate = CrossModalEvidenceCandidate(
                evidence_id=ev_id,
                query_context=query_ctx,
                optical_observation_id=opt_id,
                sar_observation_id=sar_id,
                token_identity=spatial_id,
                representation_provenance="QueryConditionedSpatialFusion"
            )
            
            candidates.append(candidate)
            
        return candidates
