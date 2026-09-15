import os
import torch
import json
from typing import Dict, Any

def save_checkpoint(
    path: str,
    text_head: torch.nn.Module,
    visual_head: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Any,
    dataset_stats: Dict[str, Any],
    epoch: int,
    loss: float
):
    """
    Serializes projection heads and reproducibility metadata.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        "text_projection_state": text_head.state_dict(),
        "visual_projection_state": visual_head.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "embedding_dim": config.embedding_dim,
        "temperature": config.temperature,
        "dataset_identifier": "BigEarthNet.txt",
        "dataset_stats": dataset_stats,
        "model_identifiers": {
            "visual": "CROMA",
            "text": "Qwen/Qwen3-4B-Instruct-2507"
        },
        "training_config": {
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "epochs": config.epochs
        },
        "seed": config.seed,
        "epoch": epoch,
        "final_loss": loss
    }
    
    torch.save(checkpoint, path)
    
    # Save config alongside
    config_path = path.replace(".pt", "_config.json")
    with open(config_path, 'w') as f:
        # Avoid dumping non-serializable objects
        json.dump(checkpoint["training_config"], f, indent=2)

def load_checkpoint(path: str, text_head: torch.nn.Module, visual_head: torch.nn.Module, map_location='cpu'):
    """
    Loads projection heads from a checkpoint.
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    text_head.load_state_dict(checkpoint["text_projection_state"])
    visual_head.load_state_dict(checkpoint["visual_projection_state"])
    return checkpoint
