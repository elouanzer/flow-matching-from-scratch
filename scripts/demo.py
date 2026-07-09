import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torchvision
import gradio as gr
import numpy as np
import yaml
from PIL import Image

from src.models.mlp import FlowMatchingMLP
from src.models.unet import MnistUNet
from src.models.dit import MnistDiT

PATH_MLP = "artefacts/00_mnist_mlp.pt"
PATH_UNET = "artefacts/01_mnist_unet.pt"
PATH_DIT = "artefacts/02_mnist_dit.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Lancement de l'interface sur : {device}")

def load_model(model_class, path, **kwargs):
    model = model_class(**kwargs).to(device)
    try:
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        return model
    except FileNotFoundError:
        return model

with open("configs/mlp.yaml") as stream:
    mlp_config = yaml.safe_load(stream)
model_mlp = load_model(FlowMatchingMLP, PATH_MLP, **mlp_config)
with open("configs/unet.yaml") as stream:
    unet_config = yaml.safe_load(stream)
model_unet = load_model(MnistUNet, PATH_UNET, **unet_config)
with open("configs/dit.yaml") as stream:
    dit_config = yaml.safe_load(stream)
model_dit = load_model(MnistDiT, PATH_DIT, **dit_config)


def run_euler(model, x_init, num_steps):
    x = x_init.clone()
    dt = 1.0 / num_steps
    
    with torch.no_grad():
        for step in range(num_steps):
            t_val = step * dt
            t_tensor = torch.full((x.shape[0],), t_val, device=device)
            
            v_pred = model(x, t_tensor)
                
            x = x + v_pred * dt
            
    return x

def process_to_pil(x_tensor):
    x_tensor = (x_tensor + 1.0) / 2.0
    x_tensor = x_tensor.clamp(0, 1)
    
    img_np = x_tensor[0, 0].cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    img = Image.fromarray(img_np, mode="L")
    return img.resize((280, 280), resample=Image.Resampling.NEAREST)

def generate_comparison(num_steps, num_images):
    x0 = torch.randn(1, 1, 28, 28).to(device)
    
    out_mlp = run_euler(model_mlp, x0, num_steps)
    out_unet = run_euler(model_unet, x0, num_steps)
    out_dit = run_euler(model_dit, x0, num_steps)
    
    return process_to_pil(out_mlp), process_to_pil(out_unet), process_to_pil(out_dit)

css = """
.large-image img { 
    width: 100% !important; 
    height: auto !important; 
    image-rendering: pixelated; 
    image-rendering: crisp-edges;
}
"""

with gr.Blocks(title="Comparaison Flow Matching") as demo:
    gr.Markdown("# Flow Matching")
    
    with gr.Row():
        steps_slider = gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Euler steps.")
        generate_btn = gr.Button("Start generation.", variant="primary")
        
    with gr.Row():
        img_mlp = gr.Image(label="1. MLP", elem_classes="large-image")
        img_unet = gr.Image(label="2. U-Net", elem_classes="large-image")
        img_dit = gr.Image(label="3. DiT", elem_classes="large-image")

    generate_btn.click(
        fn=generate_comparison,
        inputs=[steps_slider],
        outputs=[img_mlp, img_unet, img_dit]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)