# -*- coding: utf-8 -*
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class S2Attn(nn.Module):
    def __init__(self, embed_dim, num_heads, group_size):
        super(S2Attn, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.group_size = group_size

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape  # B: batch size, N: sequence length, C: embed_dim
        # Reshape x to (B, num_heads, N, head_dim)
        x = x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute query, key, value
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # Reshape for multi-head attention
        query = query.transpose(-2, -1)  # (B, num_heads, head_dim, N)
        key = key.transpose(-2, -1)    # (B, num_heads, head_dim, N)
        value = value.transpose(-2, -1)  # (B, num_heads, head_dim, N)

        # Split into groups
        num_groups = N // self.group_size
        query = query.view(B, self.num_heads, self.head_dim, num_groups, self.group_size)
        key = key.view(B, self.num_heads, self.head_dim, num_groups, self.group_size)
        value = value.view(B, self.num_heads, self.head_dim, num_groups, self.group_size)

        # Shifted attention within groups
        attn_weights = torch.matmul(query, key.transpose(-2, -1))
        attn_weights = attn_weights / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply mask to zero out the future tokens
        attn_mask = torch.ones((num_groups, self.group_size, self.group_size), device=x.device).triu(1).bool()
        attn_weights = attn_weights.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Apply shifted mask for half of the attention heads
        shifted_mask = torch.cat([attn_mask[:, :, 1:], attn_mask[:, :, :-1]], dim=-1)
        attn_weights = attn_weights.masked_fill(shifted_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Compute output
        output = torch.matmul(attn_weights, value)
        output = output.view(B, self.num_heads, self.head_dim, num_groups * self.group_size)
        output = output.transpose(1, 2).reshape(B, N, C)

        return output

# Test the S2Attn module
def test_s2attn():
    embed_dim = 256
    num_heads = 8
    group_size = 64
    batch_size = 1
    seq_length = 512

    # Create a random input tensor
    x = torch.rand(batch_size, seq_length, embed_dim)

    # Create the S2Attn module
    s2attn = S2Attn(embed_dim, num_heads, group_size)

    # Forward pass
    output = s2attn(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

test_s2attn()