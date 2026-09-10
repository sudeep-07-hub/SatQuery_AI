from typing import Dict, Any
import torch
from .input_adapter import MC4CInputAdapter
from .schema import CrossModalFusionResult, OpticalEvidence, SAREvidence, JointEvidence, SemanticMetadata
from .optical_encoder import OpticalEncoder
from .sar_encoder import SAREncoder
from .fusion import QueryConditionedFusion
from .verification import CrossModalVerifier
from .semantic_head import SemanticHead
from .utils import load_and_preprocess_image

class MC4CEngine:
    def __init__(
        self,
        croma_weights_path: str = "mc4c/weights/CROMA_base.pt",
        semantic_weights_path: str = "mc4c/weights/model.safetensors",
    ):
        self.optical_encoder = OpticalEncoder(croma_weights_path)
        self.sar_encoder = SAREncoder(croma_weights_path)
        self.fusion = QueryConditionedFusion(croma_weights_path)
        self.verifier = CrossModalVerifier()
        self.semantic_head = SemanticHead(semantic_weights_path)
        self.input_adapter = MC4CInputAdapter()

    def process(self, payload: Dict[str, Any]) -> dict:
        # 1. Validate inputs
        inputs = self.input_adapter.validate_inputs(payload)
        
        if not inputs.optical_image or not inputs.sar_image:
            raise ValueError("Cross-Modal Fusion Engine requires BOTH optical and SAR imagery. " 
                             "Silent fallbacks to single-modality processing are explicitly forbidden "
                             "under the Arch_P1_Components_AD contract.")
        
        # 2. Load images
        # S2 expects 12 channels, S1 expects 2 channels for CROMA
        opt_tensor = load_and_preprocess_image(inputs.optical_image, expected_channels=12)
        sar_tensor = load_and_preprocess_image(inputs.sar_image, expected_channels=2)
        
        # 3. Encode single modalities
        opt_out = self.optical_encoder.encode(opt_tensor)
        sar_out = self.sar_encoder.encode(sar_tensor)
        
        # 4. Fusion
        fusion_out = self.fusion(
            sar_features=torch.tensor(sar_out['patch_features']), 
            optical_features=torch.tensor(opt_out['patch_features']),
            query=inputs.query
        )
        
        # 5. Verification
        verification_out = self.verifier.verify(opt_out['global_feature'], sar_out['global_feature'])
        
        # 6. Semantic Head (Optical only in this config since BigEarthNet expects 12ch here)
        semantic_tags = self.semantic_head.get_tags(opt_tensor)[0]
        
        # 7. Assemble schema
        result = CrossModalFusionResult(
            optical_evidence=OpticalEvidence(
                source_id=inputs.optical_image,
                feature_vector=opt_out['global_feature'][0],
                spatial_features=[opt_out['patch_features'][0]] # Wrapping in extra list to match Optional[List[List[List[float]]]] or simplify schema
            ),
            sar_evidence=SAREvidence(
                source_id=inputs.sar_image,
                feature_vector=sar_out['global_feature'][0],
                spatial_features=[sar_out['patch_features'][0]]
            ),
            joint_evidence=JointEvidence(
                fused_vector=fusion_out['fused_vector'][0]
            ),
            semantic_metadata=SemanticMetadata(
                optical_tags=semantic_tags,
                sar_tags=[], # SAR semantic head omitted for simplicity or can use separate model
                joint_tags=semantic_tags # Proxy for joint
            ),
            fusion_method="CROMA_query_conditioned",
            verification_status=verification_out['verification_status'],
            confidence_score=verification_out['confidence_score']
        )
        
        return result.model_dump()
