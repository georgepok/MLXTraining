#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) for Sentiment Calibration
Based on: https://github.com/searlion/mlx-finetuning/blob/main/MLX%20LM%20GRPO.ipynb

This implementation uses TRUE policy gradients via:
1. Recomputing log probabilities from model forward passes
2. Using nn.value_and_grad() for automatic differentiation
3. Three models: trainable, rollout, and reference
4. PPO-clip objective with KL penalty
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load, generate
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner import linear_to_lora_layers
import numpy as np
import json
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict
from pathlib import Path
import tqdm

print("="*70)
print("GRPO - Group Relative Policy Optimization (v2)")
print("TRUE Policy Gradients via nn.value_and_grad()")
print("="*70)

# ============================================================================
# DATASET
# ============================================================================

@dataclass
class SentimentExample:
    text: str
    score: float
    category: str

training_examples = [
    SentimentExample("This movie was absolutely phenomenal! A masterpiece!", 0.95, "very_positive"),
    SentimentExample("Best film I've seen in years. Incredible performances!", 0.92, "very_positive"),
    SentimentExample("Stunning visuals and compelling story. Loved it!", 0.90, "very_positive"),
    SentimentExample("Exceptional in every way. Will watch again!", 0.88, "very_positive"),
    SentimentExample("Brilliant direction and powerful storytelling.", 0.93, "very_positive"),

    SentimentExample("Really enjoyed this movie. Well worth watching.", 0.80, "positive"),
    SentimentExample("Great performances and good storytelling.", 0.78, "positive"),
    SentimentExample("A solid film with memorable moments.", 0.75, "positive"),
    SentimentExample("Entertaining and well-made. Recommended.", 0.73, "positive"),
    SentimentExample("Enjoyable despite some pacing issues.", 0.71, "positive"),

    SentimentExample("Decent movie, had some good parts.", 0.62, "somewhat_positive"),
    SentimentExample("Not bad, but nothing special either.", 0.58, "somewhat_positive"),
    SentimentExample("Okay film, passes the time well enough.", 0.60, "somewhat_positive"),

    SentimentExample("The movie was fine. Neither good nor bad.", 0.50, "neutral"),
    SentimentExample("Perfectly average in every way.", 0.52, "neutral"),
    SentimentExample("Mixed feelings about this one.", 0.48, "neutral"),

    SentimentExample("Disappointing. Expected more from this.", 0.40, "somewhat_negative"),
    SentimentExample("Not great, but not terrible either.", 0.42, "somewhat_negative"),
    SentimentExample("Could have been better. Felt rushed.", 0.38, "somewhat_negative"),

    SentimentExample("Pretty bad. Would not recommend.", 0.30, "negative"),
    SentimentExample("Weak plot and poor execution.", 0.25, "negative"),
    SentimentExample("Struggled to stay interested. Not good.", 0.28, "negative"),

    SentimentExample("Terrible movie. Complete waste of time.", 0.10, "very_negative"),
    SentimentExample("One of the worst films I've ever seen.", 0.05, "very_negative"),
    SentimentExample("Awful in every possible way. Avoid!", 0.08, "very_negative"),
]

np.random.seed(42)
indices = np.random.permutation(len(training_examples))
split_idx = int(0.8 * len(training_examples))

train_set = [training_examples[i] for i in indices[:split_idx]]
val_set = [training_examples[i] for i in indices[split_idx:]]

print(f"\n📚 Dataset: {len(train_set)} train, {len(val_set)} validation")

# ============================================================================
# LOAD MODELS
# ============================================================================

print(f"\n📥 Loading models...")
model_path = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

# Load main model
model, tokenizer = load(model_path)

# Load reference model (frozen)
model_ref, _ = load(model_path)
model_ref.freeze()

print(f"✅ Models loaded")

# ============================================================================
# APPLY LORA
# ============================================================================

print(f"\n🔧 Applying LoRA...")

# LoRA config
lora_config = {
    "num_layers": 12,  # 50% of 24 layers
    "lora_parameters": {
        "rank": 8,
        "scale": 16.0,  # mlx_lm uses 'scale' not 'alpha'
        "dropout": 0.0,
    }
}

# Save LoRA config
adapter_path = Path("adapters_grpo")
adapter_path.mkdir(parents=True, exist_ok=True)
with open(adapter_path / "adapter_config.json", "w") as f:
    json.dump(lora_config, f, indent=2)

# Freeze base model and add LoRA
model.freeze()
linear_to_lora_layers(model, lora_config["num_layers"], lora_config["lora_parameters"])

# Create old model for rollouts
model_old, _ = load(model_path)
linear_to_lora_layers(model_old, lora_config["num_layers"], lora_config["lora_parameters"])
model_old.update(model.parameters())
model_old.freeze()

num_train_params = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
print(f"✅ LoRA applied: {num_train_params:,} trainable parameters")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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

def pad_sequences(sequences: List, pad_token_id: int):
    """Pad sequences to the same length"""
    if not sequences:
        return mx.array([])

    max_len = max(len(seq) for seq in sequences)
    padded_sequences = []

    for seq in sequences:
        if len(seq) < max_len:
            padding = mx.array([pad_token_id] * (max_len - len(seq)))
            padded_seq = mx.concatenate([seq, padding])
        else:
            padded_seq = seq
        padded_sequences.append(padded_seq)

    return mx.stack(padded_sequences)

# ============================================================================
# GRPO CORE FUNCTIONS
# ============================================================================

def calculate_log_probs(model, sequences, a_toks):
    """
    Calculate log probabilities for generated answer tokens.

    This is the KEY function that enables true policy gradients!

    Args:
        model: The model to evaluate
        sequences: Full sequences (prompt + answer), shape (batch, seq_len)
        a_toks: Answer tokens only, shape (batch, ans_len)

    Returns:
        Log probability of each sequence, shape (batch,)
    """
    # Forward pass to get logits
    logits = model(sequences)  # (batch, seq_len, vocab_size)

    # Convert to log probabilities
    log_probs_full = nn.log_softmax(logits, axis=-1)  # (batch, seq_len, vocab_size)

    # Find where answer tokens are in the sequence
    batch_size, seq_len = sequences.shape
    _, ans_len = a_toks.shape
    start_pos = seq_len - ans_len

    # Extract log probs for answer portion
    answer_log_probs = log_probs_full[:, start_pos:start_pos+ans_len, :]  # (batch, ans_len, vocab_size)

    # Gather log probs for actual tokens that were generated
    indices = a_toks[:, :, None]  # (batch, ans_len, 1)
    selected_log_probs = mx.take_along_axis(answer_log_probs, indices, axis=-1).squeeze(-1)  # (batch, ans_len)

    # Sum log probs across the sequence
    return mx.sum(selected_log_probs, axis=-1)  # (batch,)

def grpo_loss_fn(model, model_ref, sequences, a_toks, advantages, old_log_probs, beta, epsilon):
    """
    GRPO loss function with PPO-clip objective and KL penalty.

    Args:
        model: Trainable model (π_θ)
        model_ref: Reference model (π_ref)
        sequences: Full sequences
        a_toks: Answer tokens
        advantages: Group-relative advantages
        old_log_probs: Log probs from model_old (π_θ_old)
        beta: KL penalty coefficient
        epsilon: PPO clip parameter

    Returns:
        loss, policy_reward, kl_div
    """
    # Get log probs from trainable model
    log_probs = calculate_log_probs(model, sequences, a_toks)

    # Get log probs from reference model for KL penalty
    log_probs_ref = calculate_log_probs(model_ref, sequences, a_toks)

    # PPO-clip objective
    ratio = mx.exp(log_probs - old_log_probs)
    clipped_ratio = mx.clip(ratio, 1.0 - epsilon, 1.0 + epsilon)
    policy_reward = mx.minimum(ratio * advantages, clipped_ratio * advantages)

    # KL penalty: r - log(r) - 1, where r = π_ref / π_θ
    log_ratio_for_kl = log_probs_ref - log_probs
    ratio_for_kl = mx.exp(log_ratio_for_kl)
    kl_div = ratio_for_kl - log_ratio_for_kl - 1

    # Combined loss (maximize policy reward, minimize KL)
    loss = -mx.mean(policy_reward - beta * kl_div)

    return loss, mx.mean(policy_reward), mx.mean(kl_div)

# ============================================================================
# TRAINING LOOP
# ============================================================================

def grpo_train_loop(model, model_old, model_ref, tokenizer, optimizer, train_set, val_set,
                    iters=100, group_size=4, batch_size=2, epsilon=0.2, beta=0.01,
                    update_every=10, max_ans_len=5, eval_every=20):
    """
    GRPO training loop with rollout generation and policy optimization.
    """
    # Create gradient function using value_and_grad
    loss_and_grad_fn = nn.value_and_grad(model, grpo_loss_fn)

    history = {
        'train_loss': [],
        'policy_reward': [],
        'kl_div': [],
        'mean_reward': [],
        'val_error': [],
        'steps': []
    }

    # Baseline evaluation
    print(f"\n📊 Baseline evaluation...")
    baseline_errors = []
    for ex in val_set:
        prompt = create_prompt(ex)
        response = generate(model, tokenizer, prompt=prompt, max_tokens=max_ans_len, verbose=False)
        predicted = extract_score(response)
        error = abs(predicted - ex.score)
        baseline_errors.append(error)
    baseline_mae = np.mean(baseline_errors)
    print(f"✅ Baseline MAE: {baseline_mae:.3f}")

    best_val_error = baseline_mae

    print(f"\n🏋️ Starting GRPO training...")
    print(f"="*70)

    pbar = tqdm.tqdm(range(iters))
    for it in pbar:
        # 1. Sample batch
        indices_batch = np.random.randint(0, len(train_set), batch_size)
        batch_examples = [train_set[i] for i in indices_batch]

        # 2. Rollout: Generate G responses per prompt
        rollout_sequences = []
        rollout_rewards = []
        rollout_a_toks = []
        all_rewards = []

        for example in batch_examples:
            prompt = create_prompt(example)
            prompt_tokens = tokenizer.encode(prompt)
            group_rewards = []

            for _ in range(group_size):
                # Generate response using model_old
                response = generate(model_old, tokenizer, prompt=prompt, max_tokens=max_ans_len, verbose=False)

                # Extract answer tokens
                answer_tokens = tokenizer.encode(response, add_special_tokens=False)
                # Take only what was generated (exclude prompt)
                if len(answer_tokens) > len(prompt_tokens):
                    answer_tokens = answer_tokens[len(prompt_tokens):]

                # Compute reward
                predicted = extract_score(response)
                error = abs(predicted - example.score)
                reward = 1.0 - error  # Higher reward for lower error
                group_rewards.append(reward)
                all_rewards.append(reward)

                # Store for optimization
                full_sequence = mx.array(prompt_tokens + answer_tokens)
                rollout_sequences.append(full_sequence)
                rollout_a_toks.append(mx.array(answer_tokens))

            rollout_rewards.append(mx.array(group_rewards))

        # 3. Compute group-relative advantages (with normalization)
        advantages = []
        for rewards in rollout_rewards:
            mean_reward = mx.mean(rewards)
            std_reward = mx.sqrt(mx.var(rewards)) + 1e-8
            adv = (rewards - mean_reward) / std_reward
            advantages.append(adv)

        advantages = mx.concatenate(advantages)

        # 4. Pad sequences
        sequences = pad_sequences(rollout_sequences, tokenizer.eos_token_id)
        a_toks = pad_sequences(rollout_a_toks, tokenizer.eos_token_id)

        # 5. Calculate old log probs
        old_log_probs = calculate_log_probs(model_old, sequences, a_toks)

        # 6. Optimization step with TRUE GRADIENTS
        (loss, policy_reward, kl_div), grads = loss_and_grad_fn(
            model, model_ref, sequences, a_toks, advantages, old_log_probs, beta, epsilon
        )

        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        # 7. Record metrics
        history['train_loss'].append(loss.item())
        history['policy_reward'].append(policy_reward.item())
        history['kl_div'].append(kl_div.item())
        history['mean_reward'].append(np.mean(all_rewards))
        history['steps'].append(it + 1)

        pbar.set_description(
            f"Loss: {loss.item():.3f} | "
            f"Reward: {np.mean(all_rewards):.3f} | "
            f"KL: {kl_div.item():.4f}"
        )

        # 8. Sync model_old
        if (it + 1) % update_every == 0:
            model_old.update(model.parameters())

        # 9. Validation
        if (it + 1) % eval_every == 0:
            print(f"\n{'─'*70}")
            print(f"VALIDATION @ Step {it+1}")

            val_errors = []
            model.eval()
            for ex in val_set:
                prompt = create_prompt(ex)
                response = generate(model, tokenizer, prompt=prompt, max_tokens=max_ans_len, verbose=False)
                predicted = extract_score(response)
                error = abs(predicted - ex.score)
                val_errors.append(error)
            model.train()

            val_mae = np.mean(val_errors)
            history['val_error'].append(val_mae)

            improvement = baseline_mae - val_mae
            pct = (improvement / baseline_mae) * 100

            print(f"Val MAE: {val_mae:.3f} (Baseline: {baseline_mae:.3f})")
            print(f"Improvement: {improvement:.3f} ({pct:+.1f}%)")

            if val_mae < best_val_error:
                best_val_error = val_mae
                print(f"✅ NEW BEST!")
                # Save best model
                model.save_weights(str(adapter_path / "adapters_best.safetensors"))

            print(f"{'─'*70}\n")

    # Save final weights
    model.save_weights(str(adapter_path / "adapters.safetensors"))

    return history, best_val_error

# ============================================================================
# RUN TRAINING
# ============================================================================

# GRPO Hyperparameters
config = {
    'learning_rate': 1e-5,
    'iters': 100,
    'group_size': 4,
    'batch_size': 2,
    'epsilon': 0.2,
    'beta': 0.01,
    'update_every': 10,
    'max_ans_len': 5,
    'eval_every': 20
}

print(f"\n🎯 GRPO Configuration:")
for key, val in config.items():
    print(f"   {key}: {val}")

# Put model in training mode
model.train()

# Create optimizer
optimizer = optim.Adam(learning_rate=config['learning_rate'])

print(f"\n{'='*70}")
print("STARTING GRPO TRAINING")
print(f"{'='*70}")

# Run training
history, best_val_error = grpo_train_loop(
    model=model,
    model_old=model_old,
    model_ref=model_ref,
    tokenizer=tokenizer,
    optimizer=optimizer,
    train_set=train_set,
    val_set=val_set,
    iters=config['iters'],
    group_size=config['group_size'],
    batch_size=config['batch_size'],
    epsilon=config['epsilon'],
    beta=config['beta'],
    update_every=config['update_every'],
    max_ans_len=config['max_ans_len'],
    eval_every=config['eval_every']
)

print(f"\n{'='*70}")
print("TRAINING COMPLETE")
print(f"{'='*70}")

# Save history
with open('grpo_v2_history.json', 'w') as f:
    json.dump(history, f, indent=2)

print(f"\n✅ Saved: grpo_v2_history.json")
print(f"✅ Saved: {adapter_path}/adapters.safetensors")
print(f"✅ Saved: {adapter_path}/adapters_best.safetensors")

# Final evaluation
print(f"\n📊 FINAL RESULTS:")
baseline_errors = []
final_errors = []

model.eval()
for ex in val_set:
    prompt = create_prompt(ex)
    response = generate(model, tokenizer, prompt=prompt, max_tokens=config['max_ans_len'], verbose=False)
    predicted = extract_score(response)
    error = abs(predicted - ex.score)
    final_errors.append(error)

baseline_mae = np.mean([abs(0.5 - ex.score) for ex in val_set])  # Naive baseline
final_mae = np.mean(final_errors)

print(f"   Baseline MAE: {baseline_mae:.3f}")
print(f"   Best Val MAE: {best_val_error:.3f}")
print(f"   Final MAE: {final_mae:.3f}")

improvement = baseline_mae - best_val_error
pct = (improvement / baseline_mae) * 100
print(f"\n   Best Improvement: {improvement:.3f} ({pct:+.1f}%)")

print(f"\n{'='*70}")
print("✅ GRPO v2 WITH TRUE POLICY GRADIENTS COMPLETE")
print(f"{'='*70}")

print(f"\n🎓 Key Features:")
print(f"  ✅ TRUE policy gradients via nn.value_and_grad()")
print(f"  ✅ Log probabilities from model forward passes")
print(f"  ✅ PPO-clip objective for stable updates")
print(f"  ✅ KL penalty to prevent divergence")
print(f"  ✅ Group-relative normalized advantages")
print(f"  ✅ Three-model architecture (train, rollout, ref)")
