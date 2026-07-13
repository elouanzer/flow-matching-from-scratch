import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
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

def run_euler_trajectory(model, x_init, num_steps):
    x = x_init.clone()
    dt = 1.0 / num_steps
    trajectory = [process_to_pil(x)]
    
    with torch.no_grad():
        for step in range(num_steps):
            t_val = step * dt
            t_tensor = torch.full((x.shape[0],), t_val, device=device)
            v_pred = model(x, t_tensor)
            if v_pred.dim() == 2:
                v_pred = v_pred.view_as(x)
            x = x + v_pred * dt
            trajectory.append(process_to_pil(x))
            
    return trajectory

def generate_trajectory(model_choice, num_steps):
    if model_choice == "MLP":
        model = model_mlp
    elif model_choice == "U-Net":
        model = model_unet
    else:
        model = model_dit
        
    x0 = torch.randn(1, 1, 28, 28).to(device)
    traj_list = run_euler_trajectory(model, x0, num_steps)
    
    return traj_list, gr.update(maximum=num_steps, value=0), traj_list[0]

def update_step_image(step_index, traj_list):
    """Met à jour l'image affichée quand on bouge le curseur."""
    if not traj_list or step_index >= len(traj_list):
        return None
    return traj_list[step_index]

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

    gr.Markdown("---")  
    gr.Markdown("## Let's see the generation step by step")
    gr.Markdown("Select a model, compute the generation and then use the time cursor to actually see the generation process, from pure noise to generated image!")

    with gr.Row():
        with gr.Column():
            model_dropdown = gr.Dropdown(choices=["MLP", "U-Net", "DiT"], value="DiT", label="Model")
            traj_steps_slider = gr.Slider(minimum=2, maximum=100, value=20, step=1, label="Euler steps.")
            calc_traj_btn = gr.Button("Compute generation.", variant="secondary")
            step_scrubber = gr.Slider(minimum=0, maximum=20, value=0, step=1, label="Time t (use once generation is complete)", interactive=True)
            
        with gr.Column():
            traj_image = gr.Image(label="Image at time t", elem_classes="large-image")
            
    trajectory_state = gr.State([])

    calc_traj_btn.click(
        fn=generate_trajectory,
        inputs=[model_dropdown, traj_steps_slider],
        outputs=[trajectory_state, step_scrubber, traj_image]
    )
    
    step_scrubber.change(
        fn=update_step_image,
        inputs=[step_scrubber, trajectory_state],
        outputs=traj_image
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)