import torch
import torch.nn as nn
from encoder import Encoder
from residual_connection import ResidualConnection
from layer_normalisation import LayerNormalization

class DecoderBlock(nn.Module):
    def __init__(self, d_model, self_attention, cross_attention, feed_forward, dropout=0.1):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout)
        
        # Three residual connections for self-attention, cross-attention, and feed-forward layers
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Self-attention with residual connection
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        
        # Cross-attention with residual connection
        x = self.residual_connections[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        
        # Feed-forward network with residual connection
        x = self.residual_connections[2](x, self.feed_forward)

        return x

class Decoder(nn.Module):
    def __init__(self, features, layers, dropout_rate=0.1):
        super(Decoder, self).__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Pass the input through each layer in the decoder
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        # Apply layer normalization to the final output
        return self.norm(x)
