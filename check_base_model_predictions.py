#!/usr/bin/env python3
"""
Check what the base model actually predicts for all validation examples
"""

import numpy as np
from mlx_lm import load, generate
from dataclasses import dataclass
import re

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
    return f"""Rate this review's sentiment from 0.0 (very negative) to 1.0 (very positive).
Reply with just the number.

Review: "{example.text}"

Rating:"""

# Recreate training examples (abbreviated)
training_examples = [
    SentimentExample("This movie was absolutely phenomenal! A masterpiece!", 0.95, "very_positive"),
    SentimentExample("Best film I've seen in years. Incredible performances!", 0.92, "very_positive"),
    SentimentExample("Stunning visuals and compelling story. Loved it!", 0.90, "very_positive"),
    SentimentExample("Exceptional in every way. Will watch again!", 0.88, "very_positive"),
    SentimentExample("Brilliant direction and powerful storytelling.", 0.93, "very_positive"),
    SentimentExample("Outstanding product! Exceeded all expectations.", 0.94, "very_positive"),
    SentimentExample("The restaurant was amazing. Best meal of my life!", 0.96, "very_positive"),
    SentimentExample("Perfect service, flawless execution. Highly recommend!", 0.91, "very_positive"),
    SentimentExample("Absolutely love this! Life-changing experience.", 0.97, "very_positive"),
    SentimentExample("Incredible quality and attention to detail.", 0.89, "very_positive"),
    SentimentExample("This exceeded my wildest dreams. Pure perfection!", 0.98, "very_positive"),
    SentimentExample("Remarkable in every sense. A true gem!", 0.87, "very_positive"),
    SentimentExample("The best purchase I've ever made. Worth every penny!", 0.95, "very_positive"),
    SentimentExample("Extraordinary experience from start to finish.", 0.92, "very_positive"),
    SentimentExample("This product changed everything for me. Fantastic!", 0.90, "very_positive"),

    # Positive
    SentimentExample("Really enjoyed this movie. Well worth watching.", 0.80, "positive"),
    SentimentExample("Great performances and good storytelling.", 0.78, "positive"),
    SentimentExample("A solid film with memorable moments.", 0.75, "positive"),
    SentimentExample("Entertaining and well-made. Recommended.", 0.73, "positive"),
    SentimentExample("Enjoyable despite some pacing issues.", 0.71, "positive"),
    SentimentExample("Good quality product. Happy with my purchase.", 0.77, "positive"),
    SentimentExample("The food was delicious and service was friendly.", 0.82, "positive"),
    SentimentExample("Very pleased with the results. Would buy again.", 0.79, "positive"),
    SentimentExample("Nice experience overall. Worth the money.", 0.74, "positive"),
    SentimentExample("Impressed by the quality. Good value for price.", 0.76, "positive"),
    SentimentExample("Really good! Better than I expected.", 0.81, "positive"),
    SentimentExample("Satisfying experience. Will definitely return.", 0.72, "positive"),
    SentimentExample("Well done! Meets all my requirements.", 0.78, "positive"),
    SentimentExample("Great features and easy to use. Recommended!", 0.83, "positive"),
    SentimentExample("Positive experience overall. Very satisfied.", 0.70, "positive"),
    SentimentExample("Enjoyed it quite a bit. Good choice!", 0.76, "positive"),
    SentimentExample("Quality product with excellent performance.", 0.84, "positive"),
    SentimentExample("The staff were wonderful and accommodating.", 0.79, "positive"),

    # Somewhat Positive
    SentimentExample("Decent movie, had some good parts.", 0.62, "somewhat_positive"),
    SentimentExample("Not bad, but nothing special either.", 0.58, "somewhat_positive"),
    SentimentExample("Okay film, passes the time well enough.", 0.60, "somewhat_positive"),
    SentimentExample("The product works fine. Does what it says.", 0.61, "somewhat_positive"),
    SentimentExample("Restaurant was acceptable. Nothing extraordinary.", 0.57, "somewhat_positive"),
    SentimentExample("Pretty good for the price. Decent value.", 0.63, "somewhat_positive"),
    SentimentExample("It's okay. Has its moments but could be better.", 0.59, "somewhat_positive"),
    SentimentExample("Slightly above average. Not bad overall.", 0.64, "somewhat_positive"),
    SentimentExample("Reasonable quality. Gets the job done.", 0.56, "somewhat_positive"),
    SentimentExample("Fairly good experience. Some room for improvement.", 0.62, "somewhat_positive"),

    # Neutral
    SentimentExample("The movie was fine. Neither good nor bad.", 0.50, "neutral"),
    SentimentExample("Perfectly average in every way.", 0.52, "neutral"),
    SentimentExample("Mixed feelings about this one.", 0.48, "neutral"),
    SentimentExample("It's adequate. Nothing more, nothing less.", 0.51, "neutral"),
    SentimentExample("Standard experience. Met basic expectations.", 0.49, "neutral"),
    SentimentExample("Neutral opinion. Has pros and cons equally.", 0.50, "neutral"),
    SentimentExample("Average quality. Nothing stands out either way.", 0.53, "neutral"),
    SentimentExample("Mediocre. Neither impressed nor disappointed.", 0.47, "neutral"),
    SentimentExample("It's there. Can't say much positive or negative.", 0.54, "neutral"),
    SentimentExample("Indifferent. Wouldn't seek it out or avoid it.", 0.46, "neutral"),

    # Somewhat Negative
    SentimentExample("Disappointing. Expected more from this.", 0.40, "somewhat_negative"),
    SentimentExample("Not great, but not terrible either.", 0.42, "somewhat_negative"),
    SentimentExample("Could have been better. Felt rushed.", 0.38, "somewhat_negative"),
    SentimentExample("Below average. Had some issues throughout.", 0.41, "somewhat_negative"),
    SentimentExample("Somewhat disappointing. Missed the mark.", 0.37, "somewhat_negative"),
    SentimentExample("Not quite what I hoped for. Underwhelming.", 0.43, "somewhat_negative"),
    SentimentExample("A bit subpar. Expected higher quality.", 0.39, "somewhat_negative"),
    SentimentExample("Lackluster performance. Could use improvement.", 0.44, "somewhat_negative"),
    SentimentExample("Didn't live up to expectations. Slightly let down.", 0.36, "somewhat_negative"),
    SentimentExample("Mediocre at best. Several flaws evident.", 0.40, "somewhat_negative"),

    # Negative
    SentimentExample("Pretty bad. Would not recommend.", 0.30, "negative"),
    SentimentExample("Weak plot and poor execution.", 0.25, "negative"),
    SentimentExample("Struggled to stay interested. Not good.", 0.28, "negative"),
    SentimentExample("Poor quality. Very disappointing purchase.", 0.22, "negative"),
    SentimentExample("The service was terrible and food was cold.", 0.27, "negative"),
    SentimentExample("Bad experience. Waste of money.", 0.24, "negative"),
    SentimentExample("Really not good. Multiple problems encountered.", 0.31, "negative"),
    SentimentExample("Disappointing quality. Fell apart quickly.", 0.26, "negative"),
    SentimentExample("Unsatisfactory. Would not purchase again.", 0.29, "negative"),
    SentimentExample("Poor design and worse execution.", 0.23, "negative"),
    SentimentExample("Not worth the time or money. Regret buying.", 0.32, "negative"),
    SentimentExample("Badly made. Numerous defects found.", 0.20, "negative"),
    SentimentExample("Terrible value. Wouldn't recommend to anyone.", 0.33, "negative"),
    SentimentExample("Frustrating experience from beginning to end.", 0.21, "negative"),

    # Very Negative
    SentimentExample("Terrible movie. Complete waste of time.", 0.10, "very_negative"),
    SentimentExample("One of the worst films I've ever seen.", 0.05, "very_negative"),
    SentimentExample("Awful in every possible way. Avoid!", 0.08, "very_negative"),
    SentimentExample("Absolutely horrible. Worst purchase ever.", 0.06, "very_negative"),
    SentimentExample("Dreadful experience. Deeply regret this.", 0.09, "very_negative"),
    SentimentExample("Atrocious quality. Complete rip-off.", 0.04, "very_negative"),
    SentimentExample("Abysmal. Don't waste your time or money.", 0.11, "very_negative"),
    SentimentExample("Utterly disappointing. A complete disaster.", 0.07, "very_negative"),
    SentimentExample("Horrendous. Nothing redeeming about this.", 0.03, "very_negative"),
    SentimentExample("Pathetic quality. Worst I've ever experienced.", 0.12, "very_negative"),
    SentimentExample("Disgraceful. Avoid at all costs!", 0.02, "very_negative"),
    SentimentExample("Appalling from start to finish. Total failure.", 0.13, "very_negative"),
    SentimentExample("Catastrophic failure in every aspect.", 0.01, "very_negative"),
    SentimentExample("Execrable. I want my money back immediately.", 0.14, "very_negative"),
    SentimentExample("Nightmarish experience. Absolutely unacceptable.", 0.00, "very_negative"),
]

# Use same train/val split
np.random.seed(42)
indices = np.random.permutation(len(training_examples))
split_idx = int(0.8 * len(training_examples))
val_set = [training_examples[i] for i in indices[split_idx:]]

print("="*70)
print("BASE MODEL VALIDATION PREDICTIONS")
print("="*70)

# Load base model
print("\nLoading base model...")
model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")

errors = []
predictions = []

print(f"\nValidation predictions (19 examples):")
print("─"*70)

for i, ex in enumerate(val_set):
    prompt = create_prompt(ex)
    response = generate(model, tokenizer, prompt=prompt, max_tokens=5, verbose=False)
    predicted = extract_score(response)
    error = abs(predicted - ex.score)
    errors.append(error)
    predictions.append(predicted)

    print(f"{i+1:2d}. Target:{ex.score:.2f} | Pred:{predicted:.2f} | Error:{error:.3f} | Response:'{response}' | {ex.category}")

mae = np.mean(errors)

print("─"*70)
print(f"\n📊 Base Model MAE: {mae:.10f}")
print(f"📊 Unique predictions: {len(set(predictions))}")
print(f"📊 Prediction distribution: {set(predictions)}")

if mae > 0.387 and mae < 0.389:
    print(f"\n✅ MATCH! This is why validation MAE = {mae:.3f}")
    print(f"   The model isn't learning - still making base model predictions!")

print("\n" + "="*70)
