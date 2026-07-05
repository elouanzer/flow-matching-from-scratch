import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from src.models.mlp import FlowMatchingMLP
from src.models.layers import SinusoidalTimeEmbedding

def test_sinusoidal_embedding():
    batch_size = 16
    dim = 128
    t = torch.rand(batch_size)
    emb_layer = SinusoidalTimeEmbedding(dim=dim)
    embeddings = emb_layer(t)
    assert embeddings.shape == (batch_size, dim)

def test_mnist_time_mlp_default():
    batch_size = 8
    model = FlowMatchingMLP()
    
    x = torch.randn(batch_size, 1, 28, 28)
    t = torch.rand(batch_size)
    
    v_pred = model(x, t)
    assert v_pred.shape == x.shape