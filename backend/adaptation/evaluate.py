import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import json
import os

def compute_retrieval_metrics(image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> Dict[str, float]:
    """
    Computes R@1 and R@5 for image-to-text and text-to-image retrieval.
    Assumes image_embeddings[i] pairs with text_embeddings[i].
    """
    batch_size = image_embeddings.size(0)
    if batch_size == 0:
        return {}
        
    # [B, B] similarity matrix
    sims = torch.matmul(image_embeddings, text_embeddings.T)
    
    # image to text
    i2t_ranks = torch.zeros(batch_size)
    for i in range(batch_size):
        # descending sort of similarities
        sorted_indices = torch.argsort(sims[i], descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item()
        i2t_ranks[i] = rank
        
    # text to image
    t2i_ranks = torch.zeros(batch_size)
    for i in range(batch_size):
        sorted_indices = torch.argsort(sims[:, i], descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item()
        t2i_ranks[i] = rank
        
    metrics = {
        "i2t_r1": (i2t_ranks < 1).float().mean().item(),
        "i2t_r5": (i2t_ranks < 5).float().mean().item() if batch_size >= 5 else 1.0,
        "t2i_r1": (t2i_ranks < 1).float().mean().item(),
        "t2i_r5": (t2i_ranks < 5).float().mean().item() if batch_size >= 5 else 1.0,
        "mean_diagonal_sim": torch.diag(sims).mean().item()
    }
    
    return metrics

def generate_evaluation_report(
    config: Any, 
    dataset_stats: Dict[str, Any], 
    baseline_metrics: Dict[str, float],
    final_metrics: Dict[str, float],
    train_loss: float,
    checkpoint_path: str,
    output_dir: str
):
    """
    Dumps evaluation metrics into JSON and Markdown.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    report_dict = {
        "dataset": "BigEarthNet.txt",
        "paired_samples": dataset_stats.get("paired_records", 0),
        "train_samples": dataset_stats.get("paired_records", 0),  # Currently no validation split
        "validation_samples": 0,
        "model_ids": {
            "visual": "CROMA",
            "text": "Qwen/Qwen3-4B-Instruct-2507"
        },
        # Revisions to be populated dynamically if possible, or noted
        "embedding_dimension": config.embedding_dim,
        "projection_architecture": "Linear + L2 Normalization",
        "temperature": config.temperature,
        "optimizer": "AdamW",
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "baseline_metrics": baseline_metrics,
        "final_metrics": final_metrics,
        "final_train_loss": train_loss,
        "checkpoint": checkpoint_path,
        "reproducibility": {
            "seed": config.seed
        }
    }
    
    # Write JSON
    json_path = os.path.join(output_dir, "adaptation_report.json")
    with open(json_path, 'w') as f:
        json.dump(report_dict, f, indent=2)
        
    # Write Markdown
    md_path = os.path.join(output_dir, "adaptation_report.md")
    with open(md_path, 'w') as f:
        f.write("# Task 5.8: BigEarthNet Image-Text Contrastive Adaptation Report\n\n")
        f.write("## Dataset\n")
        f.write(f"- Paired Samples: {report_dict['paired_samples']}\n")
        f.write(f"- Train Samples: {report_dict['train_samples']}\n")
        f.write(f"- Validation Samples: {report_dict['validation_samples']} (unavailable due to dataset size)\n\n")
        
        f.write("## Training Configuration\n")
        f.write(f"- Embedding Dimension: {report_dict['embedding_dimension']}\n")
        f.write(f"- Batch Size: {report_dict['batch_size']}\n")
        f.write(f"- Epochs: {report_dict['epochs']}\n")
        f.write(f"- Learning Rate: {report_dict['learning_rate']}\n")
        f.write(f"- Temperature: {report_dict['temperature']}\n\n")
        
        f.write("## Results\n")
        f.write(f"- Final Train Loss: {report_dict['final_train_loss']:.4f}\n\n")
        
        f.write("### Baseline Metrics\n")
        for k, v in baseline_metrics.items():
            f.write(f"- {k}: {v:.4f}\n")
            
        f.write("\n### Final Metrics\n")
        for k, v in final_metrics.items():
            f.write(f"- {k}: {v:.4f}\n")
            
        f.write(f"\n## Checkpoint\n- Saved to: `{checkpoint_path}`\n")
