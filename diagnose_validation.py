#!/usr/bin/env python3
"""
Diagnose why validation MAE is constant
"""

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.tuner import linear_to_lora_layers
from mlx.utils import tree_flatten
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
    """Create sentiment scoring prompt"""
    return f"""Rate this review's sentiment from 0.0 (very negative) to 1.0 (very positive).
Reply with just the number.

Review: "{example.text}"

Rating:"""

print("="*70)
print("VALIDATION DIAGNOSIS")
print("="*70)

# Create test examples
test_examples = [
    SentimentExample("Terrible movie. Complete waste of time.", 0.10, "very_negative"),
    SentimentExample("The movie was fine. Neither good nor bad.", 0.50, "neutral"),
    SentimentExample("Best film I've seen in years. Incredible!", 0.92, "very_positive"),
]

# Load base model
print("\n1. Loading base model...")
model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
print("   ✅ Loaded")

# Test base model predictions
print("\n2. Base model predictions (before LoRA):")
print("─"*70)
for ex in test_examples:
    prompt = create_prompt(ex)
    response = generate(model, tokenizer, prompt=prompt, max_tokens=5, verbose=False)
    predicted = extract_score(response)
    error = abs(predicted - ex.score)
    print(f"Target: {ex.score:.2f} | Predicted: {predicted:.2f} | Response: '{response}' | Error: {error:.3f}")

# Apply LoRA
print("\n3. Applying LoRA...")
lora_config = {
    "num_layers": 12,
    "lora_parameters": {"rank": 8, "scale": 16.0, "dropout": 0.0}
}
model.freeze()
linear_to_lora_layers(model, lora_config["num_layers"], lora_config["lora_parameters"])
print("   ✅ LoRA applied (untrained)")

# Test with untrained LoRA
print("\n4. Predictions with untrained LoRA:")
print("─"*70)
for ex in test_examples:
    prompt = create_prompt(ex)
    response = generate(model, tokenizer, prompt=prompt, max_tokens=5, verbose=False)
    predicted = extract_score(response)
    error = abs(predicted - ex.score)
    print(f"Target: {ex.score:.2f} | Predicted: {predicted:.2f} | Response: '{response}' | Error: {error:.3f}")

# Load trained adapters if they exist
adapter_path = Path("adapters_grpo")
if (adapter_path / "adapters.safetensors").exists():
    print("\n5. Loading trained LoRA adapters...")
    model.load_weights(str(adapter_path / "adapters.safetensors"))
    print("   ✅ Adapters loaded")

    print("\n6. Predictions with TRAINED LoRA:")
    print("─"*70)
    for ex in test_examples:
        prompt = create_prompt(ex)
        response = generate(model, tokenizer, prompt=prompt, max_tokens=5, verbose=False)
        predicted = extract_score(response)
        error = abs(predicted - ex.score)
        print(f"Target: {ex.score:.2f} | Predicted: {predicted:.2f} | Response: '{response}' | Error: {error:.3f}")

    # Check if predictions changed
    print("\n7. Comparing base vs trained:")
    print("─"*70)

    # Reload base model
    model_base, _ = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")

    base_responses = []
    trained_responses = []

    for ex in test_examples:
        prompt = create_prompt(ex)

        # Base prediction
        resp_base = generate(model_base, tokenizer, prompt=prompt, max_tokens=5, verbose=False)
        base_responses.append(resp_base)

        # Trained prediction
        resp_trained = generate(model, tokenizer, prompt=prompt, max_tokens=5, verbose=False)
        trained_responses.append(resp_trained)

        if resp_base == resp_trained:
            print(f"❌ IDENTICAL: '{resp_base}' (no learning!)")
        else:
            print(f"✅ DIFFERENT: Base='{resp_base}' vs Trained='{resp_trained}'")

    if all(base == trained for base, trained in zip(base_responses, trained_responses)):
        print("\n❌ CRITICAL: All predictions IDENTICAL to base model!")
        print("   Training had ZERO effect on outputs")
    else:
        print("\n✅ Training DID change predictions")

else:
    print("\n⚠️  No trained adapters found at adapters_grpo/adapters.safetensors")
    print("   Run training first")

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)
