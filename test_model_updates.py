#!/usr/bin/env python3
"""
Test if model parameters actually change during optimization
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner import linear_to_lora_layers
from mlx.utils import tree_flatten
import copy

print("="*70)
print("MODEL UPDATE TEST")
print("="*70)

# Load and setup model
print("\n1. Setting up model...")
model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")

lora_config = {
    "num_layers": 12,
    "lora_parameters": {"rank": 8, "scale": 16.0, "dropout": 0.0}
}

model.freeze()
linear_to_lora_layers(model, lora_config["num_layers"], lora_config["lora_parameters"])

print("   ✅ Model ready")

# Save initial parameters
print("\n2. Saving initial parameter snapshot...")
initial_params = {}
for name, param in tree_flatten(model.trainable_parameters()):
    initial_params[name] = mx.array(param)  # Create copy

print(f"   Saved {len(initial_params)} parameter tensors")

# Create optimizer
optimizer = optim.Adam(learning_rate=1e-6)

# Test prompt
test_prompt = """Rate this review's sentiment from 0.0 (very negative) to 1.0 (very positive).
Reply with just the number.

Review: "This was okay, nothing special."

Rating:"""

# Get initial prediction
print("\n3. Initial model prediction...")
response_before = generate(model, tokenizer, prompt=test_prompt, max_tokens=5, verbose=False)
print(f"   Prediction: {response_before}")

# Define loss function
def simple_loss(model, sequence, a_toks, target_score):
    """Loss that encourages specific score prediction"""
    logits = model(sequence)
    log_probs = nn.log_softmax(logits, axis=-1)

    batch_size, seq_len = sequence.shape
    _, ans_len = a_toks.shape
    start_pos = seq_len - ans_len

    answer_log_probs = log_probs[:, start_pos:start_pos+ans_len, :]
    indices = a_toks[:, :, None]
    selected = mx.take_along_axis(answer_log_probs, indices, axis=-1).squeeze(-1)

    return -mx.mean(selected)  # Maximize likelihood of answer tokens

# Prepare training data
prompt_tokens = tokenizer.encode(test_prompt)
answer_tokens = tokenizer.encode("0.5", add_special_tokens=False)  # Target answer
sequence = mx.array(prompt_tokens + answer_tokens)[None, :]
a_toks = mx.array(answer_tokens)[None, :]

# Training loop
print("\n4. Running 50 optimization steps...")
loss_and_grad = nn.value_and_grad(model, simple_loss)

losses = []
for step in range(50):
    loss, grads = loss_and_grad(model, sequence, a_toks, 0.5)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    losses.append(loss.item())

    if step % 10 == 0:
        print(f"   Step {step}: Loss = {loss.item():.4f}")

print(f"   Final loss: {losses[-1]:.4f}")
print(f"   Loss change: {losses[0]:.4f} → {losses[-1]:.4f} (Δ{losses[-1]-losses[0]:+.4f})")

# Check if parameters changed
print("\n5. Checking parameter changes...")
param_changes = []
for name, param in tree_flatten(model.trainable_parameters()):
    if name in initial_params:
        diff = mx.abs(param - initial_params[name]).max()
        param_changes.append((name, float(diff)))

param_changes.sort(key=lambda x: x[1], reverse=True)

max_change = max(change for _, change in param_changes)
params_changed = sum(1 for _, change in param_changes if change > 1e-6)

print(f"   Max parameter change: {max_change:.6e}")
print(f"   Parameters changed (>1e-6): {params_changed}/{len(param_changes)}")

if max_change < 1e-8:
    print("   ❌ CRITICAL: Parameters did NOT change!")
elif max_change < 1e-6:
    print("   ⚠️  WARNING: Very small parameter changes")
else:
    print("   ✅ Parameters updated successfully")
    print(f"\n   Top 5 changed parameters:")
    for name, change in param_changes[:5]:
        print(f"      {name}: {change:.6e}")

# Get new prediction
print("\n6. Model prediction after training...")
response_after = generate(model, tokenizer, prompt=test_prompt, max_tokens=5, verbose=False)
print(f"   Prediction: {response_after}")

if response_before == response_after:
    print("   ⚠️  WARNING: Prediction unchanged (may need more training)")
else:
    print("   ✅ Prediction changed!")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

if max_change < 1e-8:
    print("\n❌ MODEL NOT UPDATING")
    print("   Parameters are frozen or optimizer is not working")
elif max_change < 1e-6:
    print("\n⚠️  TINY UPDATES")
    print("   Learning rate may be too low or gradients too small")
else:
    print("\n✅ MODEL IS UPDATING")
    print(f"   Loss decreased: {losses[0]:.4f} → {losses[-1]:.4f}")
    print(f"   Max param change: {max_change:.6e}")

    if response_before == response_after:
        print("\n   ⚠️  But predictions unchanged - may need:")
        print("      • More training steps")
        print("      • Higher learning rate")
        print("      • Different loss function")

print("="*70)
