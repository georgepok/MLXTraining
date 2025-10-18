#!/usr/bin/env python3
"""
Calculate expected MAE if model only predicts 0.0 or 1.0
"""

import numpy as np
from dataclasses import dataclass

@dataclass
class SentimentExample:
    text: str
    score: float
    category: str

# Recreate training examples
training_examples = [
    # Very Positive (0.85-1.0) - 15 examples
    *[SentimentExample("text", score, "very_positive") for score in [0.95, 0.92, 0.90, 0.88, 0.93, 0.94, 0.96, 0.91, 0.97, 0.89, 0.98, 0.87, 0.95, 0.92, 0.90]],

    # Positive (0.65-0.84) - 18 examples
    *[SentimentExample("text", score, "positive") for score in [0.80, 0.78, 0.75, 0.73, 0.71, 0.77, 0.82, 0.79, 0.74, 0.76, 0.81, 0.72, 0.78, 0.83, 0.70, 0.76, 0.84, 0.79]],

    # Somewhat Positive (0.55-0.64) - 10 examples
    *[SentimentExample("text", score, "somewhat_positive") for score in [0.62, 0.58, 0.60, 0.61, 0.57, 0.63, 0.59, 0.64, 0.56, 0.62]],

    # Neutral (0.45-0.54) - 10 examples
    *[SentimentExample("text", score, "neutral") for score in [0.50, 0.52, 0.48, 0.51, 0.49, 0.50, 0.53, 0.47, 0.54, 0.46]],

    # Somewhat Negative (0.35-0.44) - 10 examples
    *[SentimentExample("text", score, "somewhat_negative") for score in [0.40, 0.42, 0.38, 0.41, 0.37, 0.43, 0.39, 0.44, 0.36, 0.40]],

    # Negative (0.15-0.34) - 14 examples
    *[SentimentExample("text", score, "negative") for score in [0.30, 0.25, 0.28, 0.22, 0.27, 0.24, 0.31, 0.26, 0.29, 0.23, 0.32, 0.20, 0.33, 0.21]],

    # Very Negative (0.0-0.14) - 15 examples
    *[SentimentExample("text", score, "very_negative") for score in [0.10, 0.05, 0.08, 0.06, 0.09, 0.04, 0.11, 0.07, 0.03, 0.12, 0.02, 0.13, 0.01, 0.14, 0.00]],
]

# Use same train/val split
np.random.seed(42)
indices = np.random.permutation(len(training_examples))
split_idx = int(0.8 * len(training_examples))
val_set = [training_examples[i] for i in indices[split_idx:]]

print("="*70)
print("EXPECTED MAE CALCULATION")
print("="*70)

print(f"\nValidation set size: {len(val_set)}")
print(f"\nValidation examples:")
for i, ex in enumerate(val_set):
    print(f"  {i+1}. Category: {ex.category:20s} | Target: {ex.score:.2f}")

# Scenario 1: Model always predicts 0.5
print(f"\n{'─'*70}")
print("Scenario 1: Model always predicts 0.5 (baseline)")
print(f"{'─'*70}")
errors_baseline = [abs(0.5 - ex.score) for ex in val_set]
mae_baseline = np.mean(errors_baseline)
print(f"MAE: {mae_baseline:.10f}")

# Scenario 2: Model predicts 0.0 for negative, 1.0 for positive
print(f"\n{'─'*70}")
print("Scenario 2: Binary prediction (0.0 for <0.5, 1.0 for >=0.5)")
print(f"{'─'*70}")
errors_binary = []
for ex in val_set:
    # Simple rule: if sentiment is negative/neutral, predict 0.0, else 1.0
    if ex.score < 0.5:
        predicted = 0.0
    else:
        predicted = 1.0
    error = abs(predicted - ex.score)
    errors_binary.append(error)

mae_binary = np.mean(errors_binary)
print(f"MAE: {mae_binary:.10f}")

# Scenario 3: Model uses simple sentiment categories
print(f"\n{'─'*70}")
print("Scenario 3: Category-based prediction")
print(f"{'─'*70}")
category_map = {
    "very_negative": 0.0,
    "negative": 0.2,
    "somewhat_negative": 0.4,
    "neutral": 0.5,
    "somewhat_positive": 0.6,
    "positive": 0.8,
    "very_positive": 1.0
}
errors_category = [abs(category_map.get(ex.category, 0.5) - ex.score) for ex in val_set]
mae_category = np.mean(errors_category)
print(f"MAE: {mae_category:.10f}")

# Find which scenario matches 0.3878947368421053
target_mae = 0.3878947368421053
print(f"\n{'='*70}")
print(f"ACTUAL VALIDATION MAE: {target_mae:.10f}")
print(f"{'='*70}")

if abs(mae_baseline - target_mae) < 0.001:
    print(f"✅ MATCH: Baseline (always 0.5)")
    print(f"   → Model is NOT learning anything!")
elif abs(mae_binary - target_mae) < 0.001:
    print(f"✅ MATCH: Binary prediction (0.0 or 1.0)")
    print(f"   → Model learned binary classification but not calibration!")
elif abs(mae_category - target_mae) < 0.001:
    print(f"✅ MATCH: Category-based prediction")
    print(f"   → Model learned rough categories but not fine-grained scores!")
else:
    print(f"⚠️  NO CLEAR MATCH")
    print(f"   Baseline:  {mae_baseline:.10f}")
    print(f"   Binary:    {mae_binary:.10f}")
    print(f"   Category:  {mae_category:.10f}")
    print(f"   Actual:    {target_mae:.10f}")

print(f"{'='*70}")
