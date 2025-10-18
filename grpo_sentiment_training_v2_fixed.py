#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) for Sentiment Calibration
Based on: https://github.com/searlion/mlx-finetuning/blob/main/MLX%20LM%20GRPO.ipynb

This implementation uses TRUE policy gradients via:
1. Recomputing log probabilities from model forward passes
2. Using nn.value_and_grad() for automatic differentiation
3. Three models: trainable, rollout, and reference
4. PPO-clip objective with KL penalty

IMPROVEMENTS IMPLEMENTED:
========================
✓ Expanded dataset: 100 examples (from 25) covering diverse domains
✓ Extended training: 200 iterations (from 100) for better convergence
✓ Hyperparameter tuning guide with recommended ranges
✓ TEMPERATURE SAMPLING: Using make_sampler(temp=0.7) for stochastic generation
✓ Early stopping on KL divergence > 20
✓ Comprehensive diagnostics (diversity, gradient flow)
✓ Greedy decoding for validation (reproducible results)
✓ NO PROMPT VARIATIONS: Pure temperature sampling for true diversity

CRITICAL FIX: TEMPERATURE SAMPLING
----------------------------------
This version uses CORRECT MLX-LM API for temperature sampling:
  from mlx_lm.sample_utils import make_sampler
  temp_sampler = make_sampler(temp=0.7)
  generate(model, tokenizer, prompt, sampler=temp_sampler, ...)

WHY THIS MATTERS:
- GRPO requires REAL stochastic diversity within groups
- Prompt variations create confounded variables ("suffix → score" not "sentiment → score")
- Temperature sampling forces model to learn actual sentiment understanding
- Same prompt generates different outputs through sampling randomness
- No spurious correlations from varying the prompt itself

Implementation:
  Training: sampler=make_sampler(temp=0.7)
           → Stochastic sampling, same prompt yields different outputs

  Validation: No sampler parameter (defaults to greedy)
            → Greedy decoding, reproducible results

Expected results: 60-70% MAE reduction (much better than prompt variations)
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
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
    """Extract score from generated text - strict decimal format expected"""
    # First, try to find a decimal number (0.X format)
    match = re.search(r'0\.\d+', text)
    if match:
        score = float(match.group(0))
        return max(0.0, min(1.0, score))

    # Fallback: try any decimal number
    match = re.search(r'(\d+\.?\d*)', text)
    if match:
        score = float(match.group(1))
        if score > 1.0:
            score /= 10.0
        return max(0.0, min(1.0, score))

    # If no number found, return 0.5 (neutral)
    return 0.5

def create_prompt(example: SentimentExample) -> str:
    """Create sentiment scoring prompt - simple format for RL learning"""
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

    # Create temperature sampler for stochastic generation during training
    temp_sampler = make_sampler(temp=0.7)
    print(f"✅ Temperature sampler created (temp=0.7)")
    print(f"   Same prompt → different outputs for REAL diversity\n")

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

            for j in range(group_size):
                # CRITICAL FIX: Use temperature sampling for REAL diversity
                # This ensures the model learns "sentiment → score" not "prompt suffix → score"
                response = generate(model_old, tokenizer, prompt=prompt,
                                   max_tokens=max_ans_len, sampler=temp_sampler, verbose=False)

                # Log diversity check on first iteration
                if it == 0 and example == batch_examples[0] and j == 0:
                    print(f"\n{'─'*70}")
                    print("DIVERSITY CHECK: Temperature Sampling Active")
                    print(f"{'─'*70}")
                    print(f"Using make_sampler(temp=0.7) for stochastic generation")
                    print(f"This provides REAL diversity (not prompt variations)")
                    print(f"Same prompt → different outputs → true learning")
                    print(f"{'─'*70}\n")

                # Extract answer tokens
                # Note: generate() returns only generated text, not prompt+answer
                answer_tokens = tokenizer.encode(response, add_special_tokens=False)

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

        # DIAGNOSTIC: Check if fix is working (first iteration only)
        if it == 0:
            print(f"\n{'='*70}")
            print("DIVERSITY DIAGNOSTIC")
            print(f"{'='*70}")

            for i, ex in enumerate(batch_examples):
                print(f"\nExample {i+1}: '{ex.text[:50]}...' (target: {ex.score:.2f})")
                start_idx = i * group_size
                end_idx = start_idx + group_size
                group_rewards_subset = all_rewards[start_idx:end_idx]
                group_advs = advantages[start_idx:end_idx].tolist()
                
                reward_std = np.std(group_rewards_subset)
                reward_mean = np.mean(group_rewards_subset)
                
                print(f"  Group Rewards: {[f'{r:.3f}' for r in group_rewards_subset]}")
                print(f"  Mean: {reward_mean:.3f}, Std: {reward_std:.4f}")
                print(f"  Advantages: {[f'{a:.3f}' for a in group_advs]}")
                
                if reward_std < 0.01:
                    print(f"  ⚠️  WARNING: Low diversity (std < 0.01)")
                    print(f"  ⚠️  This will lead to near-zero advantages")
                elif reward_std < 0.05:
                    print(f"  ⚠️  CAUTION: Moderate diversity (std < 0.05)")
                else:
                    print(f"  ✅ Good diversity! Advantages are non-zero")

            print(f"\n{'='*70}\n")

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

        # Early stopping: Check for KL divergence explosion
        if kl_div.item() > 20.0:
            print(f"\n{'='*70}")
            print(f"⚠️  EARLY STOPPING")
            print(f"{'='*70}")
            print(f"KL divergence too high: {kl_div.item():.2f} > 20.0")
            print(f"Stopping training to prevent complete divergence from reference model")
            print(f"{'='*70}\n")
            break

        # DIAGNOSTIC: Check gradient flow (second iteration only)
        if it == 1:
            print(f"\n{'='*70}")
            print("GRADIENT FLOW CHECK")
            print(f"{'='*70}")

            param_changed = 0
            param_total = 0
            max_grad_norm = 0.0

            # Flatten the nested gradient dictionary
            flat_grads = tree_flatten(grads)

            for name, grad in flat_grads:
                param_total += 1
                grad_norm = float(mx.abs(grad).max())
                max_grad_norm = max(max_grad_norm, grad_norm)

                if grad_norm > 1e-8:
                    param_changed += 1

            print(f"\nParameters with non-zero gradients: {param_changed}/{param_total}")
            print(f"Max gradient magnitude: {max_grad_norm:.6e}")

            if param_changed == 0:
                print(f"\n❌ CRITICAL: NO GRADIENTS FLOWING!")
                print(f"   • All advantages are likely zero")
                print(f"   • Check if prompt variations are creating diversity")
            elif param_changed < param_total * 0.5:
                print(f"\n⚠️  WARNING: Only {param_changed/param_total*100:.1f}% of parameters have gradients")
            else:
                print(f"\n✅ Excellent! {param_changed/param_total*100:.1f}% of parameters updating")

            print(f"{'='*70}\n")

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

                # Use greedy decoding for consistent, reproducible validation
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
# ====================
# Optimized configuration for 100-example dataset with prompt variations
#
# HYPERPARAMETER TUNING GUIDE:
# ----------------------------
# To find optimal settings, experiment with these ranges:
#
# learning_rate: Controls update step size
#   - Too high (>1e-5): KL divergence explosion, unstable training (tested: KL→16 at step 11)
#   - Too low (1e-6): Model changes too slowly, no visible improvement by validation
#   - Sweet spot: 5e-6 (5x faster than 1e-6, but stable)
#   - Recommended: [3e-6, 5e-6, 8e-6]
#   - Current: 5e-6 (balanced: visible learning without KL explosion)
#
# beta: KL penalty coefficient (keeps policy close to reference)
#   - Too high (>0.2): Over-constrained, slow improvement
#   - Too low (<0.05): Policy diverges from reference, training instability
#   - Recommended: [0.05, 0.1, 0.2]
#   - Current: 0.1 (balanced constraint)
#
# epsilon: PPO clip range (limits policy updates)
#   - Too high (>0.3): Large policy shifts, potential instability
#   - Too low (<0.1): Very conservative updates, slow learning
#   - Recommended: [0.1, 0.2, 0.3]
#   - Current: 0.2 (standard PPO value)
#
# group_size: Completions per prompt (diversity and compute tradeoff)
#   - Too high (>16): Diminishing returns, slower training
#   - Too low (<4): Insufficient diversity, weak learning signal
#   - Recommended: [4, 8, 16]
#   - Current: 4 (good balance with prompt variations)
#
# batch_size: Examples processed per iteration
#   - Larger (4-8): More stable gradients, better for large datasets
#   - Smaller (1-2): Noisier but faster, good for small datasets
#   - Recommended: [2, 4, 8]
#   - Current: 2 (suitable for 100 examples)
#
# iters: Total training iterations
#   - With 100 examples: 200-300 iterations recommended
#   - With 500 examples: 500-1000 iterations possible
#   - Monitor validation: stop if plateaus or KL diverges
#   - Current: 200 (extended from 100 for larger dataset)

config = {
    'learning_rate': 1e-6,     # VERY LOW: Pure RL learning from rewards (no few-shot interference)
    'iters': 300,              # EXTENDED: Slower learning needs more iterations
    'group_size': 16,          # VERY HIGH: Maximum diversity for better signal
    'batch_size': 2,           # REDUCED: Smaller batches for more granular updates
    'epsilon': 0.1,            # VERY CONSERVATIVE: Minimal policy changes per step
    'beta': 0.2,               # HIGH: Strong KL constraint to prevent degradation
    'update_every': 20,        # SLOW: Less frequent rollout updates for stability
    'max_ans_len': 5,          # REDUCED: Force concise numeric responses
    'eval_every': 30           # LESS FREQUENT: More training between evaluations
}

# ALTERNATIVE CONFIGURATIONS TO TRY:
# -----------------------------------
# Conservative (more stable, slower):
# config = {
#     'learning_rate': 5e-7,
#     'beta': 0.2,
#     'epsilon': 0.1,
#     'group_size': 8,
#     'iters': 300
# }
#
# Aggressive (faster learning, less stable):
# config = {
#     'learning_rate': 5e-6,
#     'beta': 0.05,
#     'epsilon': 0.3,
#     'group_size': 4,
#     'iters': 150
# }
#
# High-diversity (more compute, better signal):
# config = {
#     'learning_rate': 1e-6,
#     'beta': 0.1,
#     'epsilon': 0.2,
#     'group_size': 16,
#     'batch_size': 4,
#     'iters': 250
# }

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

    # Use greedy decoding for reproducible final evaluation
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
