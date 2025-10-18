#!/usr/bin/env python3
"""
Simulate GRPO advantages computation to see if they're zero
"""

import mlx.core as mx
import numpy as np
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

print("="*70)
print("GRPO ADVANTAGES CHECK")
print("="*70)

# Load model
print("\n1. Loading model...")
model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
print("   ✅ Model loaded")

# Test example (from training set)
test_review = "Really enjoyed this movie. Well worth watching."
target_score = 0.80

test_prompt = f"""Rate this review's sentiment from 0.0 (very negative) to 1.0 (very positive).
Reply with just the number.

Review: "{test_review}"

Rating:"""

# Create temperature sampler
temp_sampler = make_sampler(temp=0.7)

# Generate group of responses (like GRPO does)
print(f"\n2. Generating group of 4 responses with temp=0.7...")
print(f"   Review: '{test_review}'")
print(f"   Target score: {target_score}")
print(f"\n   Responses:")

group_rewards = []
group_predictions = []

for i in range(4):
    response = generate(model, tokenizer, prompt=test_prompt,
                       max_tokens=5, sampler=temp_sampler, verbose=False)

    # Extract score
    import re
    match = re.search(r'(\d+\.?\d*)', response)
    if match:
        predicted = float(match.group(1))
        if predicted > 1.0:
            predicted /= 10.0
        predicted = max(0.0, min(1.0, predicted))
    else:
        predicted = 0.5

    error = abs(predicted - target_score)
    reward = 1.0 - error  # Higher reward for lower error

    group_rewards.append(reward)
    group_predictions.append(predicted)

    print(f"   [{i+1}] Response: '{response}' → Predicted: {predicted:.2f}, Error: {error:.3f}, Reward: {reward:.3f}")

# Compute advantages (GRPO style)
rewards_array = mx.array(group_rewards)
mean_reward = mx.mean(rewards_array)
std_reward = mx.sqrt(mx.var(rewards_array)) + 1e-8
advantages = (rewards_array - mean_reward) / std_reward

print(f"\n3. Group statistics:")
print(f"   Rewards: {[f'{r:.3f}' for r in group_rewards]}")
print(f"   Mean reward: {mean_reward.item():.3f}")
print(f"   Std reward: {std_reward.item():.4f}")
print(f"   Advantages: {[f'{a:.3f}' for a in advantages.tolist()]}")

# Check if advantages are meaningful
adv_std = float(mx.std(advantages))
adv_max = float(mx.abs(advantages).max())

print(f"\n4. Advantage analysis:")
print(f"   Advantage std: {adv_std:.4f}")
print(f"   Max |advantage|: {adv_max:.4f}")

if std_reward.item() < 0.01:
    print(f"   ❌ CRITICAL: Reward std < 0.01 - All rewards too similar!")
    print(f"   ❌ This means advantages ≈ 0 → no learning signal")
elif std_reward.item() < 0.05:
    print(f"   ⚠️  WARNING: Reward std < 0.05 - Low diversity")
else:
    print(f"   ✅ Good diversity - advantages are meaningful")

# Test multiple examples
print(f"\n5. Testing 5 different examples...")
test_cases = [
    ("Terrible movie. Complete waste of time.", 0.10),
    ("Not bad, but nothing special either.", 0.58),
    ("The movie was fine. Neither good nor bad.", 0.50),
    ("Great performances and good storytelling.", 0.78),
    ("Best film I've seen in years. Incredible!", 0.92),
]

diversity_scores = []

for review, target in test_cases:
    prompt = f"""Rate this review's sentiment from 0.0 (very negative) to 1.0 (very positive).
Reply with just the number.

Review: "{review}"

Rating:"""

    rewards = []
    for _ in range(4):
        response = generate(model, tokenizer, prompt=prompt,
                           max_tokens=5, sampler=temp_sampler, verbose=False)

        match = re.search(r'(\d+\.?\d*)', response)
        predicted = float(match.group(1)) if match else 0.5
        if predicted > 1.0:
            predicted /= 10.0
        predicted = max(0.0, min(1.0, predicted))

        error = abs(predicted - target)
        reward = 1.0 - error
        rewards.append(reward)

    reward_std = np.std(rewards)
    diversity_scores.append(reward_std)

    print(f"   '{review[:40]}...' → std={reward_std:.4f}")

avg_diversity = np.mean(diversity_scores)
print(f"\n   Average reward std across examples: {avg_diversity:.4f}")

if avg_diversity < 0.01:
    print(f"   ❌ CRITICAL: Avg diversity < 0.01")
    print(f"   ❌ Temperature sampling not providing enough variation")
elif avg_diversity < 0.05:
    print(f"   ⚠️  WARNING: Avg diversity < 0.05")
    print(f"   ⚠️  May need higher temperature or more samples")
else:
    print(f"   ✅ Good average diversity")

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

if avg_diversity < 0.01:
    print("\n❌ ADVANTAGES ARE ESSENTIALLY ZERO")
    print("   • Temperature sampling creates diversity in TEXT")
    print("   • But all texts map to SAME or VERY SIMILAR scores")
    print("   • Model is TOO CONFIDENT - always predicts similar values")
    print("\n   SOLUTION:")
    print("   • Model needs to see actual score variation in prompts")
    print("   • Or use different reward function that captures text diversity")
elif avg_diversity < 0.05:
    print("\n⚠️  WEAK LEARNING SIGNAL")
    print("   • Some diversity but not enough")
    print("   • Training will be very slow")
    print("\n   SOLUTIONS:")
    print("   • Increase temperature (try temp=1.0)")
    print("   • Increase group_size (try 8 or 16)")
else:
    print("\n✅ DIVERSITY IS GOOD")
    print("   • Issue must be elsewhere in GRPO implementation")
    print("   • Check: KL penalty, learning rate, validation logic")

print("="*70)
