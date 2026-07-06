import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from src.data.dataset import get_mnist_dataloader
from src.models.mlp import FlowMatchingMLP
from src.flow.matching import OptimalTransportFlowMatcher

def parse_args():
    parser = argparse.ArgumentParser(description="Train OT-CFM on MNIST")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--hidden_layers", type=int, default=5, help="Number of hidden layers in MLP")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save the model")
    parser.add_argument("--debug_overfit", action="store_true", help="Test the pipeline by overfitting a single batch")
    return parser.parse_args()

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Starting training on device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)
    dataloader = get_mnist_dataloader(batch_size=args.batch_size, is_train=True)
    if args.debug_overfit: # sanity check
        print("DEBUG MODE: Overfitting on a single batch...")
        single_batch = next(iter(dataloader))
        dataloader = [single_batch]
    
    model = FlowMatchingMLP(hidden_layers=args.hidden_layers).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    flow_matcher = OptimalTransportFlowMatcher()

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in progress_bar:

            x1 = batch[0].to(device)
            x0 = torch.randn_like(x1)
            
            t, x_t, v_t = flow_matcher.sample_location_and_target(x0, x1)
            v_pred = model(x_t, t)
            
            loss = F.mse_loss(v_pred, v_t) # L_CFM is just MSE
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            # gradient clipping, during training, I saw gradient exploding
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

    save_path = os.path.join(args.save_dir, "ot_cfm_mnist.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Model saved to {save_path}")

if __name__ == "__main__":
    main()