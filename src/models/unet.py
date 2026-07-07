import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from src.models.layers import SinusoidalTimeEmbedding

class UnetBlock(nn.Module):
    """
    Basic block for Unet model.

    The signal goes first through a subblock to adjust the number of channels. The time embedding is then
    added, followed by a second subblock to refine the features at a constant spatial resolution. Finally, 
    a transform layer reduces or increases the spatial size of the image.

    A subblock is Convolution + Activation + Normalization.
    
    Args:
        in_ch: input channels.
        out_ch: output channels.
        time_emb_dim: time dimension.
        up: True if currently in the second phase of the U, False if not.
    """
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1) 
            # 2 comes from the Unet: the decoder will take 2 inputs: image provided by the previous layer and by the encoder
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1) # multiplies the size by 2
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1) # will divide the size by 2
            
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1) # keeps same size
        self.bnorm1 = nn.GroupNorm(8, out_ch)
        self.bnorm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        
    def forward(self, x, t):
        """
        Args:
            x: Image tensor of shape (Batch, in_ch, H, W).
            t: Embedded time tensor of shape (Batch, time_emb_dim).
            
        Returns:
            h: input for the next Unet layer.
        """
        h = self.bnorm1(self.act(self.conv1(x))) 
        
        time_emb = self.act(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]
        h = h + time_emb
        
        h = self.bnorm2(self.act(self.conv2(h)))
        return self.transform(h)

class MnistUNet(nn.Module):
    """U-Net for MNIST.

    An image is described by a tuple following the (nb_channels, height, width) format.

    The architecture is the following:
    - an initial convolution (no time signal): (1, 28, 28) -> (32, 28, 28)
    - an encoder with 2 blocks:
        - (32, 28, 28) -> (64, 14, 14)
        - (64, 14, 14) -> (128, 7, 7)
    - a bottleneck: (128, 7, 7) -> (128, 7, 7)
    - a decoder with 2 blocks:
        - (128, 7, 7) -> (64, 14, 14)
        - (64, 14, 14) -> (32, 28, 28)
    - a final convolution (no time signal): (32, 28, 28) -> (1, 28, 28)
    
    Args:
        in_channels: input channels.
        out_channels: output_channles.
        time_emb_dim: time dimension.
        """
    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=128):
        super().__init__()
        down_channels = (32, 64, 128)
        up_channels = (128, 64, 32)

        # time encoder
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )

        # initial convolution
        self.conv0 = nn.Conv2d(in_channels, down_channels[0], 3, padding=1)

        # encoder: 28x28 -> 14x14 -> 7x7
        self.downs = nn.ModuleList([
            UnetBlock(down_channels[0], down_channels[1], time_emb_dim),
            UnetBlock(down_channels[1], down_channels[2], time_emb_dim)
        ])

        # bottleneck: 7x7
        self.bottleneck = nn.Sequential(
            nn.Conv2d(down_channels[2], down_channels[2], 3, padding=1),
            nn.GroupNorm(8, down_channels[2]),
            nn.SiLU()
        )

        # decoder: 7x7 -> 14x14 -> 28x28
        self.ups = nn.ModuleList([
            UnetBlock(up_channels[0], up_channels[1], time_emb_dim, up=True),
            UnetBlock(up_channels[1], up_channels[2], time_emb_dim, up=True)
        ])

        # final convolution
        self.output = nn.Conv2d(up_channels[2], out_channels, 1)

    def forward(self, x, t):
        """
        Args:
            x: Image tensor of shape (Batch, 1, 28, 28).
            t: Time tensor of shape (Batch,).
            
        Returns:
            v_pred: Predicted velocity field, exactly the same shape as x.
        """
        t_emb = self.time_mlp(t)
        
        x = self.conv0(x)
        
        residual_inputs = []
        for down in self.downs:
            x = down(x, t_emb)
            residual_inputs.append(x)

        x = self.bottleneck(x)

        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1) # concatenates x and the residual given by the encoder
            x = up(x, t_emb)

        return self.output(x)