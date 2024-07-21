import torch
import torch.nn as nn

# Implementation similar to the paper Attention is All You Need
class FeedForwardLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FeedForwardLayer, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
