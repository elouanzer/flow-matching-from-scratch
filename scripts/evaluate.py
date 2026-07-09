import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import torch
import yaml
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance

from src.dataset import get_mnist_dataloader
from src.models.dit import MnistDiT
from src.models.mlp import FlowMatchingMLP
from src.models.unet import MnistUNet

def process_images_for_fid(images):
    images = (images + 1.0) / 2.0
    images = images.clamp(0, 1)
    
    images = images.repeat(1, 3, 1, 1)
    
    images = (images * 255).to(torch.uint8)
    return images

def main():
    parser = argparse.ArgumentParser(description="FID score.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model to test (.pt)")
    parser.add_argument("--num_samples", type=int, default=2000, help="Number of images for the generation.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for generation.")
    parser.add_argument("--steps", type=int, default=50, help="Number of steps for Euler method (ODE Solver).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    fid = FrechetInceptionDistance(feature=64).to("cpu") # macos doesnt handle 32b
    # feature = 64, set it to 2048 to have a real result (but it is much longer)

    print("Real data extraction...")
    dataloader = get_mnist_dataloader(batch_size=args.batch_size, is_train=True)
    real_count = 0
    
    for batch in dataloader:
        real_images = batch[0].to(device)
        real_images_rgb = process_images_for_fid(real_images)
        
        fid.update(real_images_rgb.cpu(), real=True)
        
        real_count += real_images.size(0)
        if real_count >= args.num_samples:
            break

    print(f"Loading model at {args.model_path}...")
    if "dit" in args.model_path:
        with open("configs/dit.yaml") as stream:
            model_config = yaml.safe_load(stream)
        model = MnistDiT(**model_config).to(device)
    elif "unet" in args.model_path:
        with open("configs/unet.yaml") as stream:
            model_config = yaml.safe_load(stream)
        model = MnistUNet(**model_config).to(device)
    elif "mlp" in args.model_path:
        with open("configs/mlp.yaml") as stream:
            model_config = yaml.safe_load(stream)
        model = FlowMatchingMLP(**model_config).to(device)
    else:
        raise ValueError("Make sure dit, unet or mlp are in the model path. It is mandatory to initialize the right model.")
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()

    print(f"Start evaluating..")
    fake_count = 0
    dt = 1.0 / args.steps
    
    with torch.no_grad():
        with tqdm(total=args.num_samples, desc="Génération") as pbar:
            while fake_count < args.num_samples:
                current_batch_size = min(args.batch_size, args.num_samples - fake_count)
                
                x_gen = torch.randn(current_batch_size, 1, 28, 28).to(device)
                
                for step in range(args.steps):
                    t_val = step * dt
                    t_tensor = torch.full((current_batch_size,), t_val, device=device)
                    v_pred = model(x_gen, t_tensor)
                    x_gen = x_gen + v_pred * dt
                
                fake_images_rgb = process_images_for_fid(x_gen)
                fid.update(fake_images_rgb.cpu(), real=False)
                
                fake_count += current_batch_size
                pbar.update(current_batch_size)

    print("\nComputing final FID...")
    fid_score = fid.compute()
    print(f"=====================================")
    print(f" SCORE FID : {fid_score.item():.4f}")
    print(f"=====================================")

if __name__ == "__main__":
    main()