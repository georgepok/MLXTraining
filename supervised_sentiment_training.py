#!/usr/bin/env python3
"""
Supervised fine-tuning for sentiment scoring with proper teacher forcing.
Uses token-level cross-entropy loss to train the model.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load, generate
from mlx_lm.tuner import linear_to_lora_layers
from mlx.utils import tree_flatten
import numpy as np
import re
from dataclasses import dataclass
from pathlib import Path
import json
from typing import List, Tuple

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

def create_target_text(score: float) -> str:
    """Create target completion text"""
    return f" {score:.1f}"

# Create dataset
training_examples = [
    # Very Positive (0.85-1.0)
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

    # Positive (0.65-0.84)
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

    # Somewhat Positive (0.55-0.64)
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

    # Neutral (0.45-0.54)
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

    # Somewhat Negative (0.35-0.44)
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

    # Negative (0.15-0.34)
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

    # Very Negative (0.0-0.14)
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

# Train/validation split
np.random.seed(42)
indices = np.random.permutation(len(training_examples))
split_idx = int(0.8 * len(training_examples))
train_set = [training_examples[i] for i in indices[:split_idx]]
val_set = [training_examples[i] for i in indices[split_idx:]]

print("="*70)
print("SUPERVISED SENTIMENT FINE-TUNING")
print("="*70)
print(f"\nDataset: {len(train_set)} train, {len(val_set)} validation")

# Load model
print("\nLoading model...")
model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")

# Apply LoRA
print("Applying LoRA...")
lora_config = {
    "num_layers": 12,
    "lora_parameters": {"rank": 8, "scale": 16.0, "dropout": 0.0}
}
model.freeze()
linear_to_lora_layers(model, lora_config["num_layers"], lora_config["lora_parameters"])

# Count trainable parameters
trainable_params = sum(v.size for k, v in tree_flatten(model.trainable_parameters()))
print(f"Trainable parameters: {trainable_params:,}")

# Training configuration
config = {
    'learning_rate': 1e-4,
    'epochs': 20,
    'batch_size': 1,  # Process one at a time for simplicity
    'eval_every': 2,
}

print(f"\nConfiguration:")
for k, v in config.items():
    print(f"  {k}: {v}")

# Optimizer
optimizer = optim.Adam(learning_rate=config['learning_rate'])

# Training history
history = {
    'train_losses': [],
    'train_maes': [],
    'val_maes': [],
    'epochs': []
}

def evaluate(examples: List[SentimentExample], desc: str) -> float:
    """Evaluate model on examples"""
    errors = []
    for ex in examples:
        prompt = create_prompt(ex)
        response = generate(model, tokenizer, prompt=prompt, max_tokens=10, verbose=False)
        predicted = extract_score(response)
        error = abs(predicted - ex.score)
        errors.append(error)
    mae = np.mean(errors)
    print(f"  {desc} MAE: {mae:.4f}")
    return mae

def loss_fn(model, prompt_tokens, target_tokens):
    """Compute cross-entropy loss for the target tokens"""
    # Concatenate prompt and target for input
    full_input = mx.concatenate([prompt_tokens, target_tokens], axis=1)

    # Forward pass
    logits = model(full_input)

    # We want to predict target_tokens
    # Extract logits for positions where we predict target tokens
    prompt_len = prompt_tokens.shape[1]
    # Logits shape: (1, full_length, vocab_size)
    # We want logits at positions [prompt_len-1 : prompt_len + target_len - 1]
    # to predict target_tokens
    target_logits = logits[:, prompt_len-1:prompt_len+target_tokens.shape[1]-1, :]

    # Compute cross-entropy loss
    losses = nn.losses.cross_entropy(target_logits, target_tokens, reduction='none')

    return mx.mean(losses)

# Baseline evaluation
print("\nBaseline (before training):")
train_mae = evaluate(train_set, "Train")
val_mae = evaluate(val_set, "Val")

print("\n" + "="*70)
print("TRAINING")
print("="*70)

best_val_mae = float('inf')

for epoch in range(config['epochs']):
    print(f"\nEpoch {epoch + 1}/{config['epochs']}")
    print("─"*70)

    # Shuffle training data
    np.random.shuffle(train_set)

    epoch_losses = []
    epoch_maes = []

    # Training loop
    for i, ex in enumerate(train_set):
        # Create prompt and target
        prompt = create_prompt(ex)
        target = create_target_text(ex.score)

        # Tokenize
        prompt_tokens = mx.array(tokenizer.encode(prompt))[None, :]
        target_tokens = mx.array(tokenizer.encode(target, add_special_tokens=False))[None, :]

        # Compute loss and gradients
        loss_and_grad = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad(model, prompt_tokens, target_tokens)

        # Update parameters
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        epoch_losses.append(loss.item())

        # Periodically evaluate on this example
        if (i + 1) % 20 == 0:
            response = generate(model, tokenizer, prompt=prompt, max_tokens=10, verbose=False)
            predicted = extract_score(response)
            mae = abs(predicted - ex.score)
            epoch_maes.append(mae)

    avg_loss = np.mean(epoch_losses)
    avg_train_mae = np.mean(epoch_maes) if epoch_maes else train_mae

    print(f"  Train Loss: {avg_loss:.4f}")
    print(f"  Train MAE (sampled): {avg_train_mae:.4f}")

    # Validation
    if (epoch + 1) % config['eval_every'] == 0:
        print()
        train_mae = evaluate(train_set, "Train (full)")
        val_mae = evaluate(val_set, "Val")

        # Save history
        history['epochs'].append(epoch + 1)
        history['train_losses'].append(float(avg_loss))
        history['train_maes'].append(float(train_mae))
        history['val_maes'].append(float(val_mae))

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            print(f"  ✅ New best validation MAE!")

            # Save adapters
            adapter_path = Path("adapters_supervised")
            adapter_path.mkdir(exist_ok=True)
            model.save_weights(str(adapter_path / "adapters.safetensors"))
            print(f"  Saved adapters to {adapter_path}")

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print(f"\nBest validation MAE: {best_val_mae:.4f}")
print(f"Improvement: {val_set[0].__class__.__name__}")  # Dummy line

# Save history
history_path = "supervised_history.json"
with open(history_path, 'w') as f:
    json.dump(history, f, indent=2)
print(f"\nHistory saved to: {history_path}")

print("\n" + "="*70)
