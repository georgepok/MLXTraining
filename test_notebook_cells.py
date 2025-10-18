#!/usr/bin/env python
"""Test each cell of the notebook individually"""

import sys
import traceback

# Test Cell 1: Imports
print("Testing Cell 1: Imports...")
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np
    import matplotlib.pyplot as plt
    from typing import Optional, Tuple, List
    import time
    import random
    print("✅ Cell 1: All imports successful")
except Exception as e:
    print(f"❌ Cell 1 failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test Cell 2: Initialization
print("\nTesting Cell 2: Initialization...")
try:
    print("MLXTraining LoRA Fine-tuning Pipeline initialized!")
    # MLX doesn't have __version__ attribute
    print(f"Running on: {mx.default_device()}")
    
    # Set seed for reproducibility
    mx.random.seed(42)
    np.random.seed(42)
    random.seed(42)
    print("✅ Cell 2: Initialization successful")
except Exception as e:
    print(f"❌ Cell 2 failed: {e}")
    traceback.print_exc()

# Test Cell 3: LoRALinear class
print("\nTesting Cell 3: LoRALinear class...")
try:
    class LoRALinear(nn.Module):
        """
        LoRA (Low-Rank Adaptation) Linear Layer
        
        This replaces a standard linear layer with a LoRA-adapted version.
        The original weights are frozen, and we learn low-rank adaptation matrices.
        """
        
        def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
            super().__init__()
            
            # Original linear layer (frozen)
            self.linear = nn.Linear(in_features, out_features)
            
            # LoRA parameters
            self.rank = rank
            self.alpha = alpha
            self.scaling = alpha / rank
            
            # Low-rank matrices as parameters
            # A: (in_features, rank) - initialized with random normal
            # B: (rank, out_features) - initialized with zeros
            self.lora_A = mx.random.normal((in_features, rank)) * 0.01
            self.lora_B = mx.zeros((rank, out_features))
            
            # Optional dropout
            self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
            
            # Store original weights as frozen (we'll handle this differently in MLX)
            # In MLX, we'll manually exclude these from optimization
            self._frozen_weight = self.linear.weight
            self._frozen_bias = self.linear.bias if hasattr(self.linear, 'bias') else None
        
        def __call__(self, x):
            # Use frozen weights for linear transformation
            if self._frozen_bias is not None:
                result = x @ self._frozen_weight.T + self._frozen_bias
            else:
                result = x @ self._frozen_weight.T
            
            # LoRA adaptation: x @ A @ B
            lora_x = x @ self.lora_A
            if self.dropout is not None:
                lora_x = self.dropout(lora_x)
            lora_result = lora_x @ self.lora_B
            
            # Scale and add LoRA adaptation
            return result + lora_result * self.scaling

    # Test the LoRA layer
    print("Testing LoRA Linear Layer:")
    batch_size, seq_len, d_model = 4, 10, 64
    x = mx.random.normal((batch_size, seq_len, d_model))

    lora_layer = LoRALinear(d_model, d_model, rank=8, alpha=16.0)
    output = lora_layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"LoRA rank: {lora_layer.rank}")
    print(f"Scaling factor: {lora_layer.scaling}")

    # Count trainable parameters
    original_params = lora_layer.linear.weight.size + (lora_layer.linear.bias.size if hasattr(lora_layer.linear, 'bias') and lora_layer.linear.bias is not None else 0)
    lora_params = lora_layer.lora_A.size + lora_layer.lora_B.size
    print(f"Original parameters: {original_params} (frozen)")
    print(f"LoRA parameters: {lora_params} (trainable)")
    print(f"Parameter reduction: {lora_params / original_params * 100:.2f}%")
    
    print("✅ Cell 3: LoRALinear class successful")
except Exception as e:
    print(f"❌ Cell 3 failed: {e}")
    traceback.print_exc()

# Test Cell 4: Transformer classes
print("\nTesting Cell 4: Transformer classes...")
try:
    class LoRAMultiHeadAttention(nn.Module):
        """Multi-head attention with LoRA adapters"""
        
        def __init__(self, d_model: int, num_heads: int, lora_rank: int = 8, lora_alpha: float = 16.0):
            super().__init__()
            assert d_model % num_heads == 0
            
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads
            
            # LoRA-adapted projection layers
            self.q_proj = LoRALinear(d_model, d_model, rank=lora_rank, alpha=lora_alpha)
            self.k_proj = LoRALinear(d_model, d_model, rank=lora_rank, alpha=lora_alpha)
            self.v_proj = LoRALinear(d_model, d_model, rank=lora_rank, alpha=lora_alpha)
            self.o_proj = LoRALinear(d_model, d_model, rank=lora_rank, alpha=lora_alpha)
            
            self.scale = 1.0 / (self.d_k ** 0.5)
        
        def __call__(self, x, mask=None):
            batch_size, seq_len, _ = x.shape
            
            # Project to Q, K, V
            q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
            k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
            v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
            
            # Scaled dot-product attention
            scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
            
            if mask is not None:
                scores = scores + mask
            
            attn_weights = mx.softmax(scores, axis=-1)
            attn_output = attn_weights @ v
            
            # Reshape and project output
            attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
            return self.o_proj(attn_output)

    class LoRAFeedForward(nn.Module):
        """Feed-forward network with LoRA adapters"""
        
        def __init__(self, d_model: int, d_ff: int, lora_rank: int = 8, lora_alpha: float = 16.0):
            super().__init__()
            self.linear1 = LoRALinear(d_model, d_ff, rank=lora_rank, alpha=lora_alpha)
            self.linear2 = LoRALinear(d_ff, d_model, rank=lora_rank, alpha=lora_alpha)
            self.activation = nn.ReLU()
        
        def __call__(self, x):
            return self.linear2(self.activation(self.linear1(x)))

    class LoRATransformerBlock(nn.Module):
        """Transformer block with LoRA adapters"""
        
        def __init__(self, d_model: int, num_heads: int, d_ff: int, lora_rank: int = 8, lora_alpha: float = 16.0):
            super().__init__()
            self.attention = LoRAMultiHeadAttention(d_model, num_heads, lora_rank, lora_alpha)
            self.feed_forward = LoRAFeedForward(d_model, d_ff, lora_rank, lora_alpha)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        
        def __call__(self, x, mask=None):
            # Self-attention with residual connection
            attn_output = self.attention(self.norm1(x), mask)
            x = x + attn_output
            
            # Feed-forward with residual connection
            ff_output = self.feed_forward(self.norm2(x))
            x = x + ff_output
            
            return x

    class SimpleLoRATransformer(nn.Module):
        """Simple transformer model with LoRA adapters for demonstration"""
        
        def __init__(self, vocab_size: int, d_model: int = 128, num_heads: int = 8, 
                     num_layers: int = 4, d_ff: int = 512, max_seq_len: int = 128,
                     lora_rank: int = 8, lora_alpha: float = 16.0):
            super().__init__()
            
            self.d_model = d_model
            self.max_seq_len = max_seq_len
            
            # Embeddings (these will be fine-tuned normally, not with LoRA)
            self.token_embedding = nn.Embedding(vocab_size, d_model)
            self.position_embedding = nn.Embedding(max_seq_len, d_model)
            
            # Transformer blocks with LoRA
            self.blocks = [
                LoRATransformerBlock(d_model, num_heads, d_ff, lora_rank, lora_alpha)
                for _ in range(num_layers)
            ]
            
            # Output projection with LoRA
            self.output_projection = LoRALinear(d_model, vocab_size, rank=lora_rank, alpha=lora_alpha)
            
            self.norm = nn.LayerNorm(d_model)
        
        def __call__(self, input_ids, attention_mask=None):
            batch_size, seq_len = input_ids.shape
            
            # Create position ids using broadcasting
            position_ids = mx.arange(seq_len).reshape(1, seq_len)
            position_ids = mx.broadcast_to(position_ids, (batch_size, seq_len))
            
            # Embeddings
            token_emb = self.token_embedding(input_ids)
            pos_emb = self.position_embedding(position_ids)
            x = token_emb + pos_emb
            
            # Create causal mask for autoregressive generation
            if attention_mask is None:
                causal_mask = mx.triu(mx.ones((seq_len, seq_len)) * -1e9, k=1)
                attention_mask = causal_mask.reshape(1, 1, seq_len, seq_len)
            
            # Apply transformer blocks
            for block in self.blocks:
                x = block(x, attention_mask)
            
            # Final normalization and output projection
            x = self.norm(x)
            logits = self.output_projection(x)
            
            return logits

    # Create and test the model
    print("Creating LoRA Transformer Model:")
    vocab_size = 1000
    model = SimpleLoRATransformer(
        vocab_size=vocab_size,
        d_model=128,
        num_heads=8,
        num_layers=4,
        d_ff=512,
        lora_rank=8,
        lora_alpha=16.0
    )

    # Test forward pass
    batch_size, seq_len = 2, 16
    input_ids = mx.random.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(input_ids)

    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    
    print("✅ Cell 4: Transformer classes successful")
except Exception as e:
    print(f"❌ Cell 4 failed: {e}")
    traceback.print_exc()

print("\n" + "="*60)
print("Testing completed. Check above for any errors.")
print("="*60)