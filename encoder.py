import torch
import torch.nn as nn
from layer_normalisation import LayerNormalization

class Encoder(nn.Module):
    # 6 identical layers of Encoder Blocks in the paper
    def __init__(self, features, layers):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)