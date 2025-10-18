#!/usr/bin/env python3
"""
Compare supervised vs GRPO trained models side-by-side
"""

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.tuner import linear_to_lora_layers
import numpy as np
import re
from dataclasses import dataclass
from pathlib import Path

@dataclass
class SentimentExample:
    text: str
    score: float
    category: str

def extract_score(text: str) -> float:
    """Extract score from generated text"""
    match = re.search(r'(\d+\.?\d*)', text)
    if match:
        score = float(match.group(1))
        if score > 1.0:
            score /= 10.0
        return max(0.0, min(1.0, score))
    return 0.5

def create_prompt(example: SentimentExample) -> str:
    """Create sentiment scoring prompt with few-shot examples"""
    return f"""Rate sentiment from 0.0 (very negative) to 1.0 (very positive). Reply with just the decimal score.

Examples:
Review: "Absolutely terrible, worst ever!"
Score: 0.1

Review: "It was okay, nothing special."
Score: 0.5

Review: "Really good, enjoyed it!"
Score: 0.8

Review: "{example.text}"
Score:"""

# Test examples across sentiment spectrum
test_examples = [
    SentimentExample("This is the worst product I've ever bought!", 0.05, "very_negative"),
    SentimentExample("Pretty disappointing overall.", 0.30, "negative"),
    SentimentExample("It's okay, nothing special.", 0.50, "neutral"),
    SentimentExample("Really enjoyed it! Good quality.", 0.80, "positive"),
    SentimentExample("Absolutely phenomenal! Best ever!", 0.95, "very_positive"),
]

print("="*80)
print("SUPERVISED vs GRPO COMPARISON")
print("="*80)

# Load base model
print("\nLoading base model...")
base_model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")

# Apply LoRA for supervised model
print("Loading supervised model...")
supervised_model, _ = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
lora_config = {
    "num_layers": 12,
    "lora_parameters": {"rank": 8, "scale": 16.0, "dropout": 0.0}
}
supervised_model.freeze()
linear_to_lora_layers(supervised_model, lora_config["num_layers"], lora_config["lora_parameters"])

if Path("adapters_supervised/adapters.safetensors").exists():
    supervised_model.load_weights("adapters_supervised/adapters.safetensors")
    print("✅ Loaded supervised adapters")
else:
    print("❌ Supervised adapters not found!")
    exit(1)

# Apply LoRA for GRPO model
print("Loading GRPO model...")
grpo_model, _ = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
grpo_model.freeze()
linear_to_lora_layers(grpo_model, lora_config["num_layers"], lora_config["lora_parameters"])

if Path("adapters_grpo/adapters_best.safetensors").exists():
    grpo_model.load_weights("adapters_grpo/adapters_best.safetensors")
    print("✅ Loaded GRPO adapters")
else:
    print("⚠️  GRPO adapters not found, using regular adapters")
    if Path("adapters_grpo/adapters.safetensors").exists():
        grpo_model.load_weights("adapters_grpo/adapters.safetensors")

print("\n" + "="*80)
print("PREDICTIONS COMPARISON")
print("="*80)

base_errors = []
supervised_errors = []
grpo_errors = []

for i, ex in enumerate(test_examples):
    prompt = create_prompt(ex)

    # Base model prediction
    base_resp = generate(base_model, tokenizer, prompt=prompt, max_tokens=10, verbose=False)
    base_pred = extract_score(base_resp)
    base_error = abs(base_pred - ex.score)
    base_errors.append(base_error)

    # Supervised model prediction
    sup_resp = generate(supervised_model, tokenizer, prompt=prompt, max_tokens=10, verbose=False)
    sup_pred = extract_score(sup_resp)
    sup_error = abs(sup_pred - ex.score)
    supervised_errors.append(sup_error)

    # GRPO model prediction
    grpo_resp = generate(grpo_model, tokenizer, prompt=prompt, max_tokens=10, verbose=False)
    grpo_pred = extract_score(grpo_resp)
    grpo_error = abs(grpo_pred - ex.score)
    grpo_errors.append(grpo_error)

    print(f"\n{i+1}. {ex.category.upper()} (target: {ex.score:.2f})")
    print(f"   Review: \"{ex.text}\"")
    print(f"   ─" * 40)
    print(f"   Base:       {base_pred:.2f}  (error: {base_error:.3f})  Response: '{base_resp.strip()}'")
    print(f"   Supervised: {sup_pred:.2f}  (error: {sup_error:.3f})  Response: '{sup_resp.strip()}'")
    print(f"   GRPO:       {grpo_pred:.2f}  (error: {grpo_error:.3f})  Response: '{grpo_resp.strip()}'")

    # Show which is best
    best_error = min(base_error, sup_error, grpo_error)
    if sup_error == best_error:
        print(f"   ✅ BEST: Supervised")
    elif grpo_error == best_error:
        print(f"   ✅ BEST: GRPO")
    else:
        print(f"   ℹ️  BEST: Base model (neither training helped)")

print("\n" + "="*80)
print("OVERALL METRICS")
print("="*80)

base_mae = np.mean(base_errors)
supervised_mae = np.mean(supervised_errors)
grpo_mae = np.mean(grpo_errors)

print(f"\nMean Absolute Error (lower is better):")
print(f"  Base model:       {base_mae:.4f}")
print(f"  Supervised:       {supervised_mae:.4f}  ({((base_mae - supervised_mae)/base_mae*100):+.1f}%)")
print(f"  GRPO:             {grpo_mae:.4f}  ({((base_mae - grpo_mae)/base_mae*100):+.1f}%)")

print(f"\n{'─'*80}")
print("COMPARISON:")
if supervised_mae < grpo_mae:
    improvement = (grpo_mae - supervised_mae) / grpo_mae * 100
    print(f"✅ Supervised is {improvement:.1f}% better than GRPO")
else:
    improvement = (supervised_mae - grpo_mae) / supervised_mae * 100
    print(f"⚠️  GRPO is {improvement:.1f}% better than Supervised")

if supervised_mae < base_mae:
    improvement = (base_mae - supervised_mae) / base_mae * 100
    print(f"✅ Supervised improved {improvement:.1f}% over base model")
else:
    degradation = (supervised_mae - base_mae) / base_mae * 100
    print(f"❌ Supervised degraded {degradation:.1f}% from base model")

if grpo_mae < base_mae:
    improvement = (base_mae - grpo_mae) / base_mae * 100
    print(f"✅ GRPO improved {improvement:.1f}% over base model")
else:
    degradation = (grpo_mae - base_mae) / base_mae * 100
    print(f"❌ GRPO degraded {degradation:.1f}% from base model")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if supervised_mae < base_mae and supervised_mae < grpo_mae:
    print("\n🎯 SUPERVISED LEARNING WINS")
    print("   ✅ Better than base model")
    print("   ✅ Better than GRPO")
    print("   → Use adapters_supervised/ for production")
elif grpo_mae < base_mae and grpo_mae < supervised_mae:
    print("\n🎯 GRPO WINS")
    print("   ✅ Better than base model")
    print("   ✅ Better than supervised")
    print("   → Use adapters_grpo/ for production")
else:
    print("\n⚠️  NEITHER METHOD CLEARLY BETTER")
    print("   Analysis needed")

print("\n" + "="*80)
