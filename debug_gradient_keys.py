#!/usr/bin/env python3
"""
Debug gradient keys - find out what names are actually in the gradients dictionary
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.tuner import linear_to_lora_layers

print("="*70)
print("DEBUGGING GRADIENT KEYS")
print("="*70)

# Load model
print("\n1. Loading model...")
model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")

# Apply LoRA
print("\n2. Applying LoRA...")
lora_config = {
    "num_layers": 2,
    "lora_parameters": {
        "rank": 8,
        "scale": 16.0,
        "dropout": 0.0,
    }
}
model.freeze()
linear_to_lora_layers(model, lora_config["num_layers"], lora_config["lora_parameters"])

# Check trainable parameters
print("\n3. Trainable parameters:")
from mlx.utils import tree_flatten
trainable = model.trainable_parameters()
flat_trainable = tree_flatten(trainable)
print(f"Number of trainable parameter groups: {len(flat_trainable)}")
for i, (path, param) in enumerate(flat_trainable[:10]):  # First 10
    print(f"  {i}: {path} -> shape {param.shape}")

# Create simple test
print("\n4. Creating test sequences...")
dummy_seq = mx.array([[1, 2, 3, 4, 5]])
dummy_a_toks = mx.array([[4, 5]])

def calculate_log_probs(model, sequences, a_toks):
    logits = model(sequences)
    log_probs_full = nn.log_softmax(logits, axis=-1)
    batch_size, seq_len = sequences.shape
    _, ans_len = a_toks.shape
    start_pos = seq_len - ans_len
    answer_log_probs = log_probs_full[:, start_pos:start_pos+ans_len, :]
    indices = a_toks[:, :, None]
    selected_log_probs = mx.take_along_axis(answer_log_probs, indices, axis=-1).squeeze(-1)
    return mx.sum(selected_log_probs, axis=-1)

def simple_loss_fn(model, sequences, a_toks):
    log_probs = calculate_log_probs(model, sequences, a_toks)
    return -mx.mean(log_probs)

print("\n5. Computing gradients...")
loss_and_grad_fn = nn.value_and_grad(model, simple_loss_fn)
loss, grads = loss_and_grad_fn(model, dummy_seq, dummy_a_toks)

print(f"\nLoss: {loss.item():.6f}")

# Check gradient keys
print("\n6. Gradient keys:")
flat_grads = tree_flatten(grads)
print(f"Number of gradients: {len(flat_grads)}")

if len(flat_grads) == 0:
    print("\n❌ NO GRADIENTS COMPUTED!")
    print("\nThis means nn.value_and_grad() is not finding trainable parameters.")
    print("Possible issues:")
    print("  1. Model not in training mode")
    print("  2. LoRA parameters not properly registered")
    print("  3. Loss function doesn't depend on model parameters")
else:
    print("\n✅ Gradients found!")
    for i, (path, grad) in enumerate(flat_grads[:20]):  # First 20
        print(f"  {i}: {path} -> shape {grad.shape}, norm {mx.linalg.norm(grad).item():.6f}")

# Check if model is trainable
print("\n7. Model training state:")
print(f"Model frozen: {model.frozen if hasattr(model, 'frozen') else 'N/A'}")

# Try model.train()
print("\n8. Setting model to training mode...")
model.train()
print("✅ model.train() called")

print("\n9. Re-computing gradients after model.train()...")
loss2, grads2 = loss_and_grad_fn(model, dummy_seq, dummy_a_toks)
flat_grads2 = tree_flatten(grads2)
print(f"Number of gradients after model.train(): {len(flat_grads2)}")

if len(flat_grads2) > 0:
    print("\n✅ Now we have gradients!")
    for i, (path, grad) in enumerate(flat_grads2[:10]):
        print(f"  {i}: {path} -> norm {mx.linalg.norm(grad).item():.6f}")
else:
    print("\n❌ Still no gradients!")

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)
