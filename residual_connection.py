import torch
import torch.nn as nn
from layer_normalisation import LayerNormalization

class ResidualConnection(nn.Module):
    def __init__(self, features, dropout):
        super().__init__()
        self.norm = LayerNormalization(features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        return x + self.dropout(sublayer_output(self.norm(x)))
