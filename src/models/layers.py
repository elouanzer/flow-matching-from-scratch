import math
import torch
import torch.nn as nn

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        """
        Creates sinusoidal time embeddings (similar to Transformer positional encoding).
        
        Args:
            dim (int): The dimension of the output embedding. Must be an even number.
        """
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        Args:
            t: A tensor of times of shape (Batch,).
            
        Returns:
            embeddings: A tensor of shape (Batch, dim).
        """
        device = t.device
        half_dim = self.dim // 2
        
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        return embeddings