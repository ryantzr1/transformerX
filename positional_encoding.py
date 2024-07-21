import torch
import torch.nn as nn
import math

# PositionalEncoding is a module that adds positional information to the input embeddings.
# This positional information helps the model understand the position of each token in the sequence.

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        # Create a tensor to hold the positional encodings
        pe = torch.zeros(seq_len, d_model)

        # Calculate the positional encoding values
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension to make it compatible with the input embeddings
        pe = pe.unsqueeze(0)  # shape: (1, seq_len, d_model)

        # Register the positional encoding as a buffer to avoid it being considered a model parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add the positional encoding to the input embeddings
        # Detach the positional encoding to avoid gradients flowing through it
        x = x + self.pe[:, :x.shape[1], :].detach()
        return self.dropout(x)

