import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.vocab_size = vocab_size

    # Based on the formula on Section 3.4 Embeddings and Softmax of the paper
    # In the embedding layers, we multiply those weights by √dmodel.
    def forward(self, x):
        if torch.max(x) >= self.embedding.num_embeddings:
            raise IndexError(f"Token index {torch.max(x)} is out of bounds for embedding matrix with size {self.embedding.num_embeddings}")
        return self.embedding(x) * math.sqrt(self.d_model)
