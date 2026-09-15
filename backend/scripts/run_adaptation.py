import argparse
from backend.adaptation.config import AdaptationConfig
from backend.adaptation.trainer import AdaptationTrainer

def main():
    parser = argparse.ArgumentParser(description="Run Task 5.8 Image-Text Contrastive Adaptation")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    config = AdaptationConfig(
        epochs=args.epochs,
        learning_rate=args.lr
    )
    
    try:
        trainer = AdaptationTrainer(config)
        trainer.train()
    except Exception as e:
        print(f"Adaptation failed: {e}")

if __name__ == "__main__":
    main()
