import torch
import torch.nn as nn
from residual_connection import ResidualConnection
from layer_normalisation import LayerNormalization 

class EncoderBlock(nn.Module):
    def __init__(self, d_model, self_attention, feed_forward, dropout=0.1):
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        
        # Two residual connections: one for self-attention and one for the feed-forward network
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x, mask=None):
        # Self-attention with residual connection
        # The lambda function here is used to pass the output of self_attention through the residual connection.
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, mask))
        
        # Feed-forward network with residual connection
        # Directly passing the feed_forward function to the residual connection.
        x = self.residual_connections[1](x, self.feed_forward)

        return x

