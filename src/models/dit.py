import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from src.models.layers import SinusoidalTimeEmbedding

class DiTBlock(nn.Module):
    """
    Basic block for DiT model.
    
    The block uses adaptive layer normalization (adaLN-Zero) for conditioning. 
    Here is the flow of the block:
    - Modulation parameters (shift, scale, gate) are generated from context c.
    - x is normalized, scaled and shifted.
    - Self-attention is applied to x, gated, and added to the original x (residual connection).
    - x is normalized, scaled and shifted again.
    - Feed-forward (MLP) is applied, gated, and added to x (residual connection).

    Args:
        hidden_size: size of the token embeddings inside the network.
        num_heads: number of attention heads in the MHA.
    """
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size)
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c):
        """
        Args:
            x: sequence of patch embeddings of shape (Batch, num_patches, hidden_size).
            c: conditioning vector (time embedding) of shape (Batch, hidden_size).
            
        Returns:
            x: processed sequence for the next DiT layer, of shape (Batch, Seq_Len, hidden_size).
        """
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN_modulation(c).chunk(6, dim=-1)
        
        normed_x = self.norm1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        attn_out, _ = self.attn(normed_x, normed_x, normed_x)
        x = x + gate1.unsqueeze(1) * attn_out
        
        normed_x = self.norm2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        mlp_out = self.mlp(normed_x)
        x = x + gate2.unsqueeze(1) * mlp_out
        
        return x

class MnistDiT(nn.Module):
    """
    DiT for MNIST.
    
    The flow is the following:
    - the image is divided into non-overlapping patches (e.g., 4x4) and linearly projected.
    - the 2D grid of patches is flattened into a 1D sequence.
    - a learnable positional embedding is added to retain spatial awareness.
    - the sequence passes through 'depth' DiT blocks, conditioned on the time step t.
    - the final sequence is projected back to pixel space and reshaped into an image using Pixel Shuffle.

    Args:
        in_channels: number of color channels in the input image.
        patch_size: size of the square patches the image is divided into.
        hidden_size: size of the token embeddings inside the network.
        depth: number of DiTBlocks to stack.
        num_heads: number of attention heads per block.
    """
    def __init__(self, in_channels=1, patch_size=4, hidden_size=128, depth=4, num_heads=4):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        
        self.num_patches = (28 // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])
        
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.adaLN_final = nn.Linear(hidden_size, 2 * hidden_size)
        self.head = nn.Linear(hidden_size, patch_size * patch_size * in_channels)

    def forward(self, x, t):
        """
        Args:
            x: Image tensor of shape (Batch, 1, 28, 28).
            t: Time tensor of shape (Batch,).
            
        Returns:
            v_pred: Predicted velocity field, exactly the same shape as x.
        """
        x = self.patch_embed(x)
        B, _, H, W = x.shape
        
        x = x.flatten(2).transpose(1, 2)
        
        x = x + self.pos_embed
        
        c = self.time_mlp(t)
        
        for block in self.blocks:
            x = block(x, c)
            
        shift, scale = self.adaLN_final(c).chunk(2, dim=-1)
        x = self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.head(x)
        
        x = x.transpose(1, 2).view(B, -1, H, W)
        x = nn.functional.pixel_shuffle(x, self.patch_size)
        
        return x