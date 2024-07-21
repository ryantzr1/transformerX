import torch
import torch.nn as nn

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.projection1 = nn.Linear(d_model, vocab_size)
       
    def forward(self, x):
        x = self.projection1(x)
        return x
       