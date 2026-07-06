import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch.nn as nn
from src.models.layers import SinusoidalTimeEmbedding

class FlowMatchingMLP(nn.Module):
    def __init__(self, input_channels=1, image_size=28, time_emb_dim=256, hidden_dim=1024, hidden_layers=3):
        """
        A robust MLP adapted for MNIST images
        """
        super().__init__()
        self.image_dim = input_channels * image_size * image_size
        
        # time encoder
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU()
        )
        
        # spatial encoder
        self.input_mlp = nn.Sequential(
            nn.Linear(self.image_dim, hidden_dim),
            nn.SiLU()
        )
        
        # spatial decoder
        layers_mlp = []
        for _ in range(hidden_layers):
            layers_mlp.append(nn.Linear(hidden_dim, hidden_dim))
            layers_mlp.append(nn.SiLU())
        layers_mlp.append(nn.Linear(hidden_dim, self.image_dim))
        
        self.output_mlp = nn.Sequential(*layers_mlp)

    def forward(self, x, t):
        """
        Args:
            x: Image tensor of shape (Batch, 1, 28, 28).
            t: Time tensor of shape (Batch,).
            
        Returns:
            v_pred: Predicted velocity field, exactly the same shape as x.
        """
        batch_size = x.shape[0]
        
        x_flat = x.view(batch_size, -1)
        
        x_emb = self.input_mlp(x_flat)
        t_emb = self.time_mlp(t)
        
        hidden = x_emb + t_emb
        
        v_pred_flat = self.output_mlp(hidden)
        
        v_pred = v_pred_flat.view(batch_size, 1, x.shape[2], x.shape[3])
        
        return v_pred