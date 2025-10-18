#!/usr/bin/env python3
"""
Diagnose why GRPO is not learning (Val MAE = Baseline MAE = 0.388)
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner import linear_to_lora_layers
from mlx.utils import tree_flatten

print("="*70)
print("GRPO FAILURE DIAGNOSIS")
print("="*70)

# Load model
print("\n1. Loading model...")
model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
print("   ✅ Model loaded")

# Apply LoRA
print("\n2. Applying LoRA...")
lora_config = {
    "num_layers": 12,
    "lora_parameters": {
        "rank": 8,
        "scale": 16.0,
        "dropout": 0.0,
    }
}

model.freeze()
linear_to_lora_layers(model, lora_config["num_layers"], lora_config["lora_parameters"])

# Check trainable parameters
trainable_params = model.trainable_parameters()
flat_params = tree_flatten(trainable_params)
num_trainable = sum(v.size for _, v in flat_params)

print(f"   Trainable parameters: {num_trainable:,}")
print(f"   Number of trainable tensors: {len(list(flat_params))}")

if num_trainable == 0:
    print("   ❌ CRITICAL: NO TRAINABLE PARAMETERS!")
else:
    print("   ✅ LoRA parameters are trainable")

# Test temperature sampling
print("\n3. Testing temperature sampling diversity...")
test_prompt = """Rate this review's sentiment from 0.0 (very negative) to 1.0 (very positive).
Reply with just the number.

Review: "This was absolutely fantastic! Loved every minute."

Rating:"""

temp_sampler = make_sampler(temp=0.7)
print("   Created sampler: make_sampler(temp=0.7)")

samples = []
print("\n   Generating 5 samples with temperature sampling:")
for i in range(5):
    response = generate(model, tokenizer, prompt=test_prompt,
                       max_tokens=5, sampler=temp_sampler, verbose=False)
    samples.append(response)
    print(f"   Sample {i+1}: {response}")

unique = len(set(samples))
print(f"\n   Unique samples: {unique}/5")

if unique == 1:
    print("   ❌ CRITICAL: All samples IDENTICAL - temperature not working!")
elif unique < 3:
    print("   ⚠️  WARNING: Low diversity - temperature may not be effective")
else:
    print("   ✅ Temperature sampling is working")

# Test greedy decoding
print("\n4. Testing greedy decoding (should be identical)...")
greedy_samples = []
print("   Generating 3 samples without sampler (greedy):")
for i in range(3):
    response = generate(model, tokenizer, prompt=test_prompt,
                       max_tokens=5, verbose=False)
    greedy_samples.append(response)
    print(f"   Sample {i+1}: {response}")

unique_greedy = len(set(greedy_samples))
print(f"\n   Unique samples: {unique_greedy}/3")

if unique_greedy == 1:
    print("   ✅ Greedy decoding is deterministic")
else:
    print("   ⚠️  WARNING: Greedy should be deterministic!")

# Test gradient computation
print("\n5. Testing gradient computation...")

# Create simple prompt
prompt_tokens = tokenizer.encode(test_prompt)
answer_tokens = tokenizer.encode("1.0", add_special_tokens=False)

# Create sequence
sequence = mx.array(prompt_tokens + answer_tokens)[None, :]  # Add batch dim
a_toks = mx.array(answer_tokens)[None, :]  # Add batch dim

print(f"   Sequence shape: {sequence.shape}")
print(f"   Answer tokens shape: {a_toks.shape}")

# Define simple loss function
def simple_loss(model, sequence, a_toks):
    """Compute simple loss to test if gradients flow"""
    logits = model(sequence)
    log_probs = nn.log_softmax(logits, axis=-1)

    # Get log probs for answer tokens
    batch_size, seq_len = sequence.shape
    _, ans_len = a_toks.shape
    start_pos = seq_len - ans_len

    answer_log_probs = log_probs[:, start_pos:start_pos+ans_len, :]
    indices = a_toks[:, :, None]
    selected = mx.take_along_axis(answer_log_probs, indices, axis=-1).squeeze(-1)

    return -mx.mean(selected)  # Negative log likelihood

# Compute gradients
loss_and_grad = nn.value_and_grad(model, simple_loss)
loss, grads = loss_and_grad(model, sequence, a_toks)

print(f"   Loss: {loss.item():.4f}")

# Check gradients
flat_grads = tree_flatten(grads)
grads_with_values = [(name, float(mx.abs(grad).max())) for name, grad in flat_grads]
nonzero_grads = [(name, val) for name, val in grads_with_values if val > 1e-8]

print(f"   Total gradient tensors: {len(list(flat_grads))}")
print(f"   Non-zero gradients: {len(nonzero_grads)}")

if len(nonzero_grads) == 0:
    print("   ❌ CRITICAL: NO GRADIENTS FLOWING!")
else:
    print(f"   ✅ Gradients are flowing")
    print(f"\n   Sample gradients:")
    for name, val in nonzero_grads[:5]:
        print(f"      {name}: {val:.6e}")

# Summary
print("\n" + "="*70)
print("DIAGNOSIS SUMMARY")
print("="*70)

issues = []

if num_trainable == 0:
    issues.append("❌ NO TRAINABLE PARAMETERS - LoRA not applied correctly")

if unique == 1:
    issues.append("❌ TEMPERATURE SAMPLING NOT WORKING - All outputs identical")
elif unique < 3:
    issues.append("⚠️  LOW SAMPLING DIVERSITY - Temperature effect weak")

if len(nonzero_grads) == 0:
    issues.append("❌ NO GRADIENT FLOW - Optimization won't work")

if len(issues) == 0:
    print("\n✅ All basic checks passed!")
    print("\nPossible issues with GRPO implementation:")
    print("  • Advantages might be zero despite diversity")
    print("  • KL penalty might be too strong")
    print("  • Learning rate might be too low")
    print("  • Reward function might not discriminate well")
else:
    print("\n⚠️  CRITICAL ISSUES FOUND:")
    for issue in issues:
        print(f"  {issue}")

print("\n" + "="*70)
