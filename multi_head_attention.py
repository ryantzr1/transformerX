import torch
import math

import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads # divide embedding into num_heads parts
        
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query_len = query.size(1)
        key_len = key.size(1)
        
        # Linear projections
        query = self.query_linear(query).view(batch_size, query_len, self.num_heads, self.d_k).transpose(1, 2)
        key = self.key_linear(key).view(batch_size, key_len, self.num_heads, self.d_k).transpose(1, 2)
        value = self.value_linear(value).view(batch_size, key_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(self.dropout(attention_weights), value)

        # Reshape and concatenate attention heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, query_len, self.num_heads * self.d_k)
        
        # Linear transformation for output
        output = self.output_linear(attention_output)
        
        return output
