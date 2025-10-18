"""
LoRA (Low-Rank Adaptation) Models for MLX
Provides LoRA implementations for use with RLVR and other training approaches
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional


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
        
        # Low-rank matrices
        # A: (in_features, rank) - initialized with random normal
        # B: (rank, out_features) - initialized with zeros
        self.lora_A = mx.random.normal((in_features, rank)) * 0.01
        self.lora_B = mx.zeros((rank, out_features))
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        
        # Freeze original linear layer parameters
        # In MLX, we'll handle freezing at the training level
        self._frozen_params = ['linear.weight']
        if hasattr(self.linear, 'bias') and self.linear.bias is not None:
            self._frozen_params.append('linear.bias')
    
    def __call__(self, x):
        # Original linear transformation
        result = self.linear(x)
        
        # LoRA adaptation: x @ A @ B
        lora_x = x @ self.lora_A
        if self.dropout is not None:
            lora_x = self.dropout(lora_x)
        lora_result = lora_x @ self.lora_B
        
        # Scale and add LoRA adaptation
        return result + lora_result * self.scaling


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
        
        # Manually compute softmax for MLX compatibility
        max_scores = mx.max(scores, axis=-1, keepdims=True)
        shifted_scores = scores - max_scores
        exp_scores = mx.exp(shifted_scores)
        attn_weights = exp_scores / mx.sum(exp_scores, axis=-1, keepdims=True)
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
        
        # Create position ids
        position_ids = mx.broadcast_to(mx.arange(seq_len).reshape(1, seq_len), (batch_size, seq_len))
        
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
    
    def named_parameters(self):
        """Get all named parameters (for compatibility)"""
        params = []
        
        # Embeddings (trainable)
        params.append(("token_embedding.weight", self.token_embedding.weight))
        params.append(("position_embedding.weight", self.position_embedding.weight))
        
        # Transformer blocks - LoRA parameters
        for i, block in enumerate(self.blocks):
            # Attention LoRA params
            params.append((f"blocks.{i}.attention.q_proj.lora_A", block.attention.q_proj.lora_A))
            params.append((f"blocks.{i}.attention.q_proj.lora_B", block.attention.q_proj.lora_B))
            params.append((f"blocks.{i}.attention.k_proj.lora_A", block.attention.k_proj.lora_A))
            params.append((f"blocks.{i}.attention.k_proj.lora_B", block.attention.k_proj.lora_B))
            params.append((f"blocks.{i}.attention.v_proj.lora_A", block.attention.v_proj.lora_A))
            params.append((f"blocks.{i}.attention.v_proj.lora_B", block.attention.v_proj.lora_B))
            params.append((f"blocks.{i}.attention.o_proj.lora_A", block.attention.o_proj.lora_A))
            params.append((f"blocks.{i}.attention.o_proj.lora_B", block.attention.o_proj.lora_B))
            
            # Feed-forward LoRA params
            params.append((f"blocks.{i}.feed_forward.linear1.lora_A", block.feed_forward.linear1.lora_A))
            params.append((f"blocks.{i}.feed_forward.linear1.lora_B", block.feed_forward.linear1.lora_B))
            params.append((f"blocks.{i}.feed_forward.linear2.lora_A", block.feed_forward.linear2.lora_A))
            params.append((f"blocks.{i}.feed_forward.linear2.lora_B", block.feed_forward.linear2.lora_B))
            
            # Layer norm parameters (trainable)
            params.append((f"blocks.{i}.norm1.weight", block.norm1.weight))
            params.append((f"blocks.{i}.norm1.bias", block.norm1.bias))
            params.append((f"blocks.{i}.norm2.weight", block.norm2.weight))
            params.append((f"blocks.{i}.norm2.bias", block.norm2.bias))
        
        # Output projection LoRA params
        params.append(("output_projection.lora_A", self.output_projection.lora_A))
        params.append(("output_projection.lora_B", self.output_projection.lora_B))
        
        # Final norm parameters (trainable)
        params.append(("norm.weight", self.norm.weight))
        params.append(("norm.bias", self.norm.bias))
        
        return params
    
    def parameters(self):
        """Get all parameters"""
        return [param for _, param in self.named_parameters()]
    
    def trainable_parameters(self):
        """Get only trainable parameters (LoRA + embeddings + layer norms)"""
        return self.parameters()  # All our parameters are trainable