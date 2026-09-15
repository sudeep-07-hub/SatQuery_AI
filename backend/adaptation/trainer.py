import torch
from torch.utils.data import DataLoader
import os

from backend.adaptation.config import AdaptationConfig
from backend.adaptation.dataset import BigEarthNetAdaptationDataset
from backend.adaptation.projections import TextProjectionHead, VisualProjectionHead
from backend.adaptation.contrastive import SymmetricInfoNCE
from backend.adaptation.evaluate import compute_retrieval_metrics, generate_evaluation_report
from backend.adaptation.checkpoint import save_checkpoint

class AdaptationTrainer:
    def __init__(self, config: AdaptationConfig):
        self.config = config
        self.device = config.device
        
        # Set seeds for reproducibility
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
            
        print(f"Initializing AdaptationTrainer on device {self.device}")
        
        self.dataset = BigEarthNetAdaptationDataset(
            visual_dir=config.features_dir,
            text_dir=config.text_features_dir,
            required_sample_ids=["771130"]  # Constrained to the known sample for STAGE A smoke test
        )
        
        if len(self.dataset) == 0:
            raise RuntimeError("Dataset is empty. STAGE A requires sample 771130.")
            
        self.dataloader = DataLoader(self.dataset, batch_size=config.batch_size, shuffle=True)
        
        self.text_head = TextProjectionHead(input_dim=config.text_dim, output_dim=config.embedding_dim).to(self.device)
        self.visual_head = VisualProjectionHead(input_dim=config.visual_dim, output_dim=config.embedding_dim).to(self.device)
        self.loss_fn = SymmetricInfoNCE(temperature=config.temperature).to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            list(self.text_head.parameters()) + list(self.visual_head.parameters()),
            lr=config.learning_rate
        )

    def evaluate_batch(self, batch) -> tuple[float, dict]:
        self.text_head.eval()
        self.visual_head.eval()
        
        with torch.no_grad():
            v_feats = batch["visual_features"].to(self.device)
            t_feats = batch["text_features"].to(self.device)
            
            v_emb = self.visual_head(v_feats)
            t_emb = self.text_head(t_feats)
            
            loss = self.loss_fn(v_emb, t_emb)
            metrics = compute_retrieval_metrics(v_emb, t_emb)
            
        return loss.item(), metrics

    def get_baseline_metrics(self) -> dict:
        print("Computing baseline metrics (untrained projections)...")
        # In a real setup with a validation loader, we would evaluate over the whole validation set.
        # Since we only have 1 sample, we evaluate on the training loader.
        batch = next(iter(self.dataloader))
        loss, metrics = self.evaluate_batch(batch)
        metrics["loss"] = loss
        return metrics
        
    def train(self):
        print(f"Starting STAGE A Adaptation Training. Dataset size: {len(self.dataset)}")
        baseline_metrics = self.get_baseline_metrics()
        print(f"Baseline: {baseline_metrics}")
        
        final_loss = 0.0
        
        for epoch in range(self.config.epochs):
            self.text_head.train()
            self.visual_head.train()
            
            epoch_loss = 0.0
            
            for batch in self.dataloader:
                v_feats = batch["visual_features"].to(self.device)
                t_feats = batch["text_features"].to(self.device)
                
                self.optimizer.zero_grad()
                
                v_emb = self.visual_head(v_feats)
                t_emb = self.text_head(t_feats)
                
                loss = self.loss_fn(v_emb, t_emb)
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(self.dataloader)
            final_loss = avg_loss
            print(f"Epoch {epoch+1}/{self.config.epochs} | Loss: {avg_loss:.4f}")
            
        # Final evaluation
        print("Computing final metrics...")
        batch = next(iter(self.dataloader))
        final_eval_loss, final_metrics = self.evaluate_batch(batch)
        final_metrics["loss"] = final_eval_loss
        print(f"Final Metrics: {final_metrics}")
        
        checkpoint_path = os.path.join(self.config.checkpoints_dir, "bigearthnet_qwen3_croma_adapter.pt")
        save_checkpoint(
            path=checkpoint_path,
            text_head=self.text_head,
            visual_head=self.visual_head,
            optimizer=self.optimizer,
            config=self.config,
            dataset_stats=self.dataset.stats,
            epoch=self.config.epochs,
            loss=final_loss
        )
        print(f"Checkpoint saved to {checkpoint_path}")
        
        generate_evaluation_report(
            config=self.config,
            dataset_stats=self.dataset.stats,
            baseline_metrics=baseline_metrics,
            final_metrics=final_metrics,
            train_loss=final_loss,
            checkpoint_path=checkpoint_path,
            output_dir=self.config.metrics_dir
        )
        print(f"Reports saved to {self.config.metrics_dir}")
