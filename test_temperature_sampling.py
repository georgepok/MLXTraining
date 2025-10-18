#!/usr/bin/env python3
"""
Quick test to verify mlx_lm built-in temperature sampling works correctly.
"""

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

print("="*70)
print("Testing MLX-LM Built-in Temperature Sampling")
print("="*70)

print("\n📥 Loading model...")
model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
print("✅ Model loaded")

test_prompt = """Rate this review's sentiment from 0.0 (very negative) to 1.0 (very positive).
Reply with just the number.

Review: "This was absolutely fantastic! Loved every minute."

Rating:"""

print(f"\n🎲 Test 1: Generating 10 samples with temp=0.7")
print(f"   (Same prompt → different outputs proves stochastic sampling)")
print("─"*70)

# Create temperature sampler (CORRECT API)
temp_sampler = make_sampler(temp=0.7)

samples_temp = []
for i in range(10):
    response = generate(model, tokenizer, prompt=test_prompt, max_tokens=5, sampler=temp_sampler, verbose=False)
    samples_temp.append(response)
    print(f"Sample {i+1}: {response}")

print("─"*70)

# Check for diversity
unique_samples = len(set(samples_temp))
print(f"\n📊 Diversity Check (temp=0.7):")
print(f"   Unique samples: {unique_samples}/10")

if unique_samples >= 3:
    print(f"   ✅ EXCELLENT: Temperature sampling is working!")
    print(f"   ✅ Multiple different outputs from same prompt")
elif unique_samples == 1:
    print(f"   ❌ FAILED: All samples identical (not using temperature)")
    print(f"   ❌ Temperature parameter may not be working")
else:
    print(f"   ⚠️  WARNING: Low diversity ({unique_samples} unique)")
    print(f"   ⚠️  May need to increase temperature or check implementation")

# Test greedy decoding
print(f"\n🎯 Test 2: Generating 5 samples with greedy decoding (no temp)")
print(f"   (Should be identical for reproducibility)")
print("─"*70)

samples_greedy = []
for i in range(5):
    response = generate(model, tokenizer, prompt=test_prompt, max_tokens=5, verbose=False)
    samples_greedy.append(response)
    print(f"Sample {i+1}: {response}")

print("─"*70)

unique_greedy = len(set(samples_greedy))
print(f"\n📊 Consistency Check (greedy):")
print(f"   Unique samples: {unique_greedy}/5")

if unique_greedy == 1:
    print(f"   ✅ PERFECT: All samples identical (reproducible)")
else:
    print(f"   ⚠️  WARNING: {unique_greedy} unique samples in greedy mode")
    print(f"   ⚠️  Greedy decoding should be deterministic")

print("\n" + "="*70)
print("🎯 SUMMARY")
print("="*70)
if unique_samples >= 3 and unique_greedy == 1:
    print("✅ Temperature sampling: WORKING")
    print("✅ Greedy decoding: WORKING")
    print("✅ Ready for GRPO training!")
else:
    print("⚠️  Issues detected - review output above")
print("="*70)
