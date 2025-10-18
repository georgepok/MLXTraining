#!/usr/bin/env python3
"""
GRPO Implementation Aligned with searlion/mlx-finetuning Notebook
Key innovation: Group-relative normalization for stable training
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load, generate
from mlx_lm.tuner import linear_to_lora_layers
from mlx_lm.sample_utils import make_sampler
from mlx.utils import tree_flatten
import numpy as np
import re
from dataclasses import dataclass
from pathlib import Path
import json
from typing import List

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

def compute_reward(predicted_score: float, target_score: float) -> float:
    """Compute reward (negative MAE)"""
    return -abs(predicted_score - target_score)

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

    # Positive (0.65-0.84)
    SentimentExample("Really enjoyed this movie. Well worth watching.", 0.80, "positive"),
    SentimentExample("Great performances and good storytelling.", 0.78, "positive"),
    SentimentExample("A solid film with memorable moments.", 0.75, "positive"),
    SentimentExample("Entertaining and well-made. Recommended.", 0.73, "positive"),
    SentimentExample("Good quality product. Happy with my purchase.", 0.77, "positive"),
    SentimentExample("The food was delicious and service was friendly.", 0.82, "positive"),
    SentimentExample("Very pleased with the results. Would buy again.", 0.79, "positive"),
    SentimentExample("Nice experience overall. Worth the money.", 0.74, "positive"),

    # Somewhat Positive (0.55-0.64)
    SentimentExample("Decent movie, had some good parts.", 0.62, "somewhat_positive"),
    SentimentExample("Not bad, but nothing special either.", 0.58, "somewhat_positive"),
    SentimentExample("Okay film, passes the time well enough.", 0.60, "somewhat_positive"),
    SentimentExample("The product works fine. Does what it says.", 0.61, "somewhat_positive"),
    SentimentExample("Pretty good for the price. Decent value.", 0.63, "somewhat_positive"),

    # Neutral (0.45-0.54)
    SentimentExample("The movie was fine. Neither good nor bad.", 0.50, "neutral"),
    SentimentExample("Perfectly average in every way.", 0.52, "neutral"),
    SentimentExample("Mixed feelings about this one.", 0.48, "neutral"),
    SentimentExample("It's adequate. Nothing more, nothing less.", 0.51, "neutral"),
    SentimentExample("Standard experience. Met basic expectations.", 0.49, "neutral"),

    # Somewhat Negative (0.35-0.44)
    SentimentExample("Disappointing. Expected more from this.", 0.40, "somewhat_negative"),
    SentimentExample("Not great, but not terrible either.", 0.42, "somewhat_negative"),
    SentimentExample("Could have been better. Felt rushed.", 0.38, "somewhat_negative"),
    SentimentExample("Below average. Had some issues throughout.", 0.41, "somewhat_negative"),
    SentimentExample("Somewhat disappointing. Missed the mark.", 0.37, "somewhat_negative"),

    # Negative (0.15-0.34)
    SentimentExample("Pretty bad. Would not recommend.", 0.30, "negative"),
    SentimentExample("Weak plot and poor execution.", 0.25, "negative"),
    SentimentExample("Struggled to stay interested. Not good.", 0.28, "negative"),
    SentimentExample("Poor quality. Very disappointing purchase.", 0.22, "negative"),
    SentimentExample("The service was terrible and food was cold.", 0.27, "negative"),
    SentimentExample("Bad experience. Waste of money.", 0.24, "negative"),

    # Very Negative (0.0-0.14)
    SentimentExample("Terrible movie. Complete waste of time.", 0.10, "very_negative"),
    SentimentExample("One of the worst films I've ever seen.", 0.05, "very_negative"),
    SentimentExample("Awful in every possible way. Avoid!", 0.08, "very_negative"),
    SentimentExample("Absolutely horrible. Worst purchase ever.", 0.06, "very_negative"),
    SentimentExample("Dreadful experience. Deeply regret this.", 0.09, "very_negative"),
]

# Train/validation split
np.random.seed(42)
indices = np.random.permutation(len(training_examples))
split_idx = int(0.8 * len(training_examples))
train_set = [training_examples[i] for i in indices[:split_idx]]
val_set = [training_examples[i] for i in indices[split_idx:]]

print("="*70)
print("GRPO - searlion/mlx-finetuning Approach")
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

trainable_params = sum(v.size for k, v in tree_flatten(model.trainable_parameters()))
print(f"Trainable parameters: {trainable_params:,}")

# Configuration based on research
config = {
    'learning_rate': 5e-6,      # Conservative for RL
    'iters': 100,               # Fewer iterations
    'group_size': 4,            # searlion uses 4
    'batch_size': 2,            # Small batch
    'beta': 0.1,                # KL penalty
    'max_ans_len': 5,           # Short answers
    'eval_every': 10,           # Frequent eval
    'temperature': 0.7,         # Production range (0.6-0.9)
}

print(f"\nConfiguration (aligned with searlion):")
for k, v in config.items():
    print(f"  {k}: {v}")

# Optimizer
optimizer = optim.Adam(learning_rate=config['learning_rate'])

# Create sampler with production temperature
sampler = make_sampler(temp=config['temperature'])
print(f"\n✅ Created sampler with temperature={config['temperature']}")

# Training history
history = {
    'train_losses': [],
    'val_maes': [],
    'iterations': []
}

def evaluate(examples: List[SentimentExample], desc: str) -> float:
    """Evaluate model"""
    errors = []
    for ex in examples:
        prompt = create_prompt(ex)
        response = generate(model, tokenizer, prompt=prompt, max_tokens=config['max_ans_len'], verbose=False)
        predicted = extract_score(response)
        error = abs(predicted - ex.score)
        errors.append(error)
    mae = np.mean(errors)
    print(f"  {desc} MAE: {mae:.4f}")
    return mae

# Baseline
print("\nBaseline evaluation:")
val_mae = evaluate(val_set, "Val")
baseline_mae = val_mae

print("\n" + "="*70)
print("STARTING GRPO TRAINING (searlion approach)")
print("="*70)

best_val_mae = float('inf')

for iteration in range(config['iters']):
    # Sample batch
    batch_examples = np.random.choice(train_set, size=config['batch_size'], replace=False)

    batch_loss = 0.0

    for example in batch_examples:
        prompt = create_prompt(example)
        prompt_tokens = mx.array(tokenizer.encode(prompt))

        # Generate GROUP_SIZE responses and collect their data
        group_data = []

        for _ in range(config['group_size']):
            # Generate response (for reward computation only)
            response = generate(
                model, tokenizer,
                prompt=prompt,
                max_tokens=config['max_ans_len'],
                temp=config['temperature'],
                verbose=False
            )

            # Extract score and compute reward
            predicted_score = extract_score(response)
            reward = compute_reward(predicted_score, example.score)

            # Get response tokens
            response_tokens = mx.array(tokenizer.encode(response, add_special_tokens=False))

            group_data.append({
                'response': response,
                'reward': reward,
                'response_tokens': response_tokens
            })

        # KEY INNOVATION: Group-relative normalization
        # A_i = (r_i - mean(r_group)) / (std(r_group) + eps)
        group_rewards = [d['reward'] for d in group_data]
        mean_reward = float(np.mean(group_rewards))
        std_reward = float(np.std(group_rewards)) + 1e-8

        # Compute normalized advantages
        advantages = [(r - mean_reward) / std_reward for r in group_rewards]

        # Compute loss for this group using policy gradients
        for data, advantage in zip(group_data, advantages):
            response_tokens = data['response_tokens']

            # Compute log probabilities for this response
            full_tokens = mx.concatenate([prompt_tokens, response_tokens])

            # Forward pass
            logits = model(full_tokens[None, :])

            # Get logits for response positions
            prompt_len = len(prompt_tokens)
            response_logits = logits[0, prompt_len-1:prompt_len+len(response_tokens)-1, :]

            # Compute log probabilities
            log_probs = nn.log_softmax(response_logits, axis=-1)

            # Get log prob of actual tokens
            token_log_probs = log_probs[mx.arange(len(response_tokens)), response_tokens]

            # Policy gradient: -log_prob * advantage
            policy_loss = -mx.sum(token_log_probs) * advantage
            batch_loss += policy_loss

    # Average loss over batch and group_size
    batch_loss = batch_loss / (config['batch_size'] * config['group_size'])

    # Compute gradients and update
    def loss_fn(params):
        # Recompute forward passes for gradient calculation
        total_loss = 0.0

        for example in batch_examples:
            prompt = create_prompt(example)
            prompt_tokens = mx.array(tokenizer.encode(prompt))

            # Re-generate to get fresh responses
            group_data = []
            for _ in range(config['group_size']):
                response = generate(
                    model, tokenizer,
                    prompt=prompt,
                    max_tokens=config['max_ans_len'],
                    temp=config['temperature'],
                    verbose=False
                )
                predicted_score = extract_score(response)
                reward = compute_reward(predicted_score, example.score)
                response_tokens = mx.array(tokenizer.encode(response, add_special_tokens=False))
                group_data.append({'reward': reward, 'response_tokens': response_tokens})

            # Group normalization
            group_rewards = [d['reward'] for d in group_data]
            mean_reward = float(np.mean(group_rewards))
            std_reward = float(np.std(group_rewards)) + 1e-8
            advantages = [(r - mean_reward) / std_reward for r in group_rewards]

            # Compute loss
            for data, advantage in zip(group_data, advantages):
                response_tokens = data['response_tokens']
                full_tokens = mx.concatenate([prompt_tokens, response_tokens])
                logits = model(full_tokens[None, :])
                prompt_len = len(prompt_tokens)
                response_logits = logits[0, prompt_len-1:prompt_len+len(response_tokens)-1, :]
                log_probs = nn.log_softmax(response_logits, axis=-1)
                token_log_probs = log_probs[mx.arange(len(response_tokens)), response_tokens]
                policy_loss = -mx.sum(token_log_probs) * advantage
                total_loss += policy_loss

        return total_loss / (config['batch_size'] * config['group_size'])

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad(model.trainable_parameters())

    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

    history['train_losses'].append(float(loss))

    # Evaluation
    if (iteration + 1) % config['eval_every'] == 0:
        print(f"\n{'─'*70}")
        print(f"Iteration {iteration + 1}/{config['iters']}")
        print(f"  Train Loss: {loss:.4f}")

        val_mae = evaluate(val_set, "Val")
        improvement = (baseline_mae - val_mae) / baseline_mae * 100

        history['iterations'].append(iteration + 1)
        history['val_maes'].append(float(val_mae))

        print(f"  Improvement: {improvement:+.1f}%")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            print(f"  ✅ New best!")

            # Save adapters
            adapter_path = Path("adapters_searlion")
            adapter_path.mkdir(exist_ok=True)
            model.save_weights(str(adapter_path / "adapters.safetensors"))

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print(f"\nBaseline MAE:  {baseline_mae:.4f}")
print(f"Best Val MAE:  {best_val_mae:.4f}")
print(f"Improvement:   {((baseline_mae - best_val_mae)/baseline_mae*100):+.1f}%")

# Save history
history_path = "searlion_history.json"
with open(history_path, 'w') as f:
    json.dump(history, f, indent=2)
print(f"\nHistory saved to: {history_path}")

print("\n" + "="*70)
