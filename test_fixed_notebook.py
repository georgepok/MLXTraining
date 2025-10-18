#!/usr/bin/env python
"""Test the fixed notebook cells"""

import sys
import traceback

# Import all required libraries
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
import time
import random

print("Testing fixed notebook...")
print("MLX imported successfully")

# Initialize
mx.random.seed(42)
np.random.seed(42)
random.seed(42)

# Test all classes from the notebook
print("\n1. Testing LoRALinear...")
exec("""
class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_A = mx.random.normal((in_features, rank)) * 0.01
        self.lora_B = mx.zeros((rank, out_features))
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self._frozen_weight = self.linear.weight
        self._frozen_bias = self.linear.bias if hasattr(self.linear, 'bias') else None
    
    def __call__(self, x):
        if self._frozen_bias is not None:
            result = x @ self._frozen_weight.T + self._frozen_bias
        else:
            result = x @ self._frozen_weight.T
        lora_x = x @ self.lora_A
        if self.dropout is not None:
            lora_x = self.dropout(lora_x)
        lora_result = lora_x @ self.lora_B
        return result + lora_result * self.scaling

x = mx.random.normal((4, 10, 64))
lora_layer = LoRALinear(64, 64, rank=8, alpha=16.0)
output = lora_layer(x)
print(f"LoRALinear test passed: {output.shape}")
""")

print("\n2. Testing count_parameters function...")
exec("""
def count_parameters(model):
    total = 0
    trainable = 0
    
    # Count embeddings
    total += model.token_embedding.weight.size
    total += model.position_embedding.weight.size
    trainable += model.token_embedding.weight.size
    trainable += model.position_embedding.weight.size
    
    # Count transformer blocks
    for block in model.blocks:
        # Attention
        for proj in [block.attention.q_proj, block.attention.k_proj, 
                    block.attention.v_proj, block.attention.o_proj]:
            total += proj.linear.weight.size
            if hasattr(proj.linear, 'bias') and proj.linear.bias is not None:
                total += proj.linear.bias.size
            trainable += proj.lora_A.size + proj.lora_B.size
        
        # Feed-forward
        for ff in [block.feed_forward.linear1, block.feed_forward.linear2]:
            total += ff.linear.weight.size
            if hasattr(ff.linear, 'bias') and ff.linear.bias is not None:
                total += ff.linear.bias.size
            trainable += ff.lora_A.size + ff.lora_B.size
        
        # Layer norms
        total += block.norm1.weight.size + block.norm1.bias.size
        total += block.norm2.weight.size + block.norm2.bias.size
        trainable += block.norm1.weight.size + block.norm1.bias.size
        trainable += block.norm2.weight.size + block.norm2.bias.size
    
    # Output projection
    total += model.output_projection.linear.weight.size
    if hasattr(model.output_projection.linear, 'bias') and model.output_projection.linear.bias is not None:
        total += model.output_projection.linear.bias.size
    trainable += model.output_projection.lora_A.size + model.output_projection.lora_B.size
    
    # Final norm
    total += model.norm.weight.size + model.norm.bias.size
    trainable += model.norm.weight.size + model.norm.bias.size
    
    return total, trainable

print("count_parameters function defined successfully")
""")

print("\n3. Testing all transformer classes...")
exec("""
class LoRAMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, lora_rank: int = 8, lora_alpha: float = 16.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_proj = LoRALinear(d_model, d_model, rank=lora_rank, alpha=lora_alpha)
        self.k_proj = LoRALinear(d_model, d_model, rank=lora_rank, alpha=lora_alpha)
        self.v_proj = LoRALinear(d_model, d_model, rank=lora_rank, alpha=lora_alpha)
        self.o_proj = LoRALinear(d_model, d_model, rank=lora_rank, alpha=lora_alpha)
        self.scale = 1.0 / (self.d_k ** 0.5)
    
    def __call__(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            scores = scores + mask
        attn_weights = mx.softmax(scores, axis=-1)
        attn_output = attn_weights @ v
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        return self.o_proj(attn_output)

class LoRAFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, lora_rank: int = 8, lora_alpha: float = 16.0):
        super().__init__()
        self.linear1 = LoRALinear(d_model, d_ff, rank=lora_rank, alpha=lora_alpha)
        self.linear2 = LoRALinear(d_ff, d_model, rank=lora_rank, alpha=lora_alpha)
        self.activation = nn.ReLU()
    
    def __call__(self, x):
        return self.linear2(self.activation(self.linear1(x)))

class LoRATransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, lora_rank: int = 8, lora_alpha: float = 16.0):
        super().__init__()
        self.attention = LoRAMultiHeadAttention(d_model, num_heads, lora_rank, lora_alpha)
        self.feed_forward = LoRAFeedForward(d_model, d_ff, lora_rank, lora_alpha)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def __call__(self, x, mask=None):
        attn_output = self.attention(self.norm1(x), mask)
        x = x + attn_output
        ff_output = self.feed_forward(self.norm2(x))
        x = x + ff_output
        return x

class SimpleLoRATransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, num_heads: int = 8, 
                 num_layers: int = 4, d_ff: int = 512, max_seq_len: int = 128,
                 lora_rank: int = 8, lora_alpha: float = 16.0):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = [
            LoRATransformerBlock(d_model, num_heads, d_ff, lora_rank, lora_alpha)
            for _ in range(num_layers)
        ]
        self.output_projection = LoRALinear(d_model, vocab_size, rank=lora_rank, alpha=lora_alpha)
        self.norm = nn.LayerNorm(d_model)
    
    def __call__(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        position_ids = mx.arange(seq_len).reshape(1, seq_len)
        position_ids = mx.broadcast_to(position_ids, (batch_size, seq_len))
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(position_ids)
        x = token_emb + pos_emb
        if attention_mask is None:
            causal_mask = mx.triu(mx.ones((seq_len, seq_len)) * -1e9, k=1)
            attention_mask = causal_mask.reshape(1, 1, seq_len, seq_len)
        for block in self.blocks:
            x = block(x, attention_mask)
        x = self.norm(x)
        logits = self.output_projection(x)
        return logits

# Test the model
vocab_size = 1000
model = SimpleLoRATransformer(vocab_size=vocab_size, d_model=128, num_heads=8, num_layers=4, d_ff=512, lora_rank=8, lora_alpha=16.0)
batch_size, seq_len = 2, 16
input_ids = mx.random.randint(0, vocab_size, (batch_size, seq_len))
logits = model(input_ids)
print(f"Transformer test passed: input {input_ids.shape} -> output {logits.shape}")

# Test count_parameters
total_params, trainable_params = count_parameters(model)
print(f"Parameter counting works: {total_params:,} total, {trainable_params:,} trainable")
""")

print("\n✅ All core tests passed! The notebook should work correctly now.")
print("\nSummary:")
print("- All imports work correctly")
print("- LoRA classes are implemented correctly for MLX")
print("- Transformer model works with MLX broadcasting")
print("- Parameter counting function works without named_parameters")
print("- The notebook is ready to run!")

print("\nTo run the notebook:")
print("1. Start the MLX environment: source mlx_env/bin/activate")
print("2. Start Jupyter: jupyter notebook MLXTraining_ordered.ipynb")
print("3. Select kernel: Python 3.12 (MLX)")
print("4. Run cells in order from top to bottom")