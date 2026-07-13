import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from src.models.mlp import FlowMatchingMLP
from src.models.unet import UnetBlock, MnistUNet
from src.models.dit import DiTBlock, MnistDiT
from src.models.layers import SinusoidalTimeEmbedding

# ====================================
# Sinusoidal Embeddings
# ====================================

def test_sinusoidal_embedding():
    batch_size = 16
    dim = 128
    t = torch.rand(batch_size)
    emb_layer = SinusoidalTimeEmbedding(dim=dim)
    embeddings = emb_layer(t)
    assert embeddings.shape == (batch_size, dim)

# ====================================
# MLP
# ====================================

def test_mnist_time_mlp_default():
    batch_size = 8
    model = FlowMatchingMLP()
    
    x = torch.randn(batch_size, 1, 28, 28)
    t = torch.rand(batch_size)
    
    v_pred = model(x, t)
    assert v_pred.shape == x.shape

# ====================================
# UNet
# ====================================

def test_block_down():
    batch_size = 4
    in_ch = 32
    out_ch = 64
    time_emb_dim = 128
    H, W = 28, 28

    block = UnetBlock(in_ch, out_ch, time_emb_dim, up=False)
    
    x = torch.randn(batch_size, in_ch, H, W)
    t_emb = torch.randn(batch_size, time_emb_dim)
    
    out = block(x, t_emb)
    
    assert out.shape == (batch_size, out_ch, H // 2, W // 2), f"Shape mismatch: {out.shape}"


def test_block_up():
    batch_size = 4
    in_ch = 64
    out_ch = 32
    time_emb_dim = 128
    H, W = 14, 14

    block = UnetBlock(in_ch, out_ch, time_emb_dim, up=True)
    
    x = torch.randn(batch_size, 2 * in_ch, H, W)
    t_emb = torch.randn(batch_size, time_emb_dim)
    
    out = block(x, t_emb)
    
    assert out.shape == (batch_size, out_ch, H * 2, W * 2), f"Shape mismatch: {out.shape}"


def test_mnist_unet_default():
    batch_size = 8
    model = MnistUNet()
    
    x = torch.randn(batch_size, 1, 28, 28)
    t = torch.rand(batch_size)
    
    v_pred = model(x, t)
    
    assert v_pred.shape == x.shape, f"Expected {x.shape}, got {v_pred.shape}"

# ====================================
# DiT
# ====================================

def test_dit_block():
    batch_size = 4
    seq_len = 49
    hidden_size = 128
    num_heads = 4

    block = DiTBlock(hidden_size, num_heads)
    
    x = torch.randn(batch_size, seq_len, hidden_size)
    t_emb = torch.randn(batch_size, hidden_size)
    
    out = block(x, t_emb)
    
    assert out.shape == (batch_size, seq_len, hidden_size), f"Shape mismatch: {out.shape}"

def test_mnist_dit_default():
    batch_size = 8
    model = MnistDiT(in_channels=1, patch_size=4, hidden_size=128, depth=2, num_heads=4)
    x = torch.randn(batch_size, 1, 28, 28)
    t = torch.rand(batch_size)
    v_pred = model(x, t)
    assert v_pred.shape == x.shape, f"Expected {x.shape}, got {v_pred.shape}"