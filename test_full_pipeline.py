#!/usr/bin/env python3
"""
Test the full GRPO pipeline end-to-end with detailed logging
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load, generate
from mlx_lm.tuner import linear_to_lora_layers
import numpy as np
import re
from dataclasses import dataclass

@dataclass
class SentimentExample:
    text: str
    score: float
    category: str

# Single test example
test_data = [
    SentimentExample("This movie was amazing!", 0.90, "very_positive"),
    SentimentExample("Pretty bad movie.", 0.30, "negative"),
]

print("="*70)
print("TESTING FULL GRPO PIPELINE")
print("="*70)

# Load models
print("\n1. Loading models...")
model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
model_ref, _ = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
model_ref.freeze()
print("✅ Models loaded")

# Apply LoRA
print("\n2. Applying LoRA...")
lora_config = {
    "num_layers": 2,
    "lora_parameters": {
        "rank": 8,
        "scale": 16.0,
        "dropout": 0.0,
    }
}
model.freeze()
linear_to_lora_layers(model, lora_config["num_layers"], lora_config["lora_parameters"])

model_old, _ = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
linear_to_lora_layers(model_old, lora_config["num_layers"], lora_config["lora_parameters"])
model_old.update(model.parameters())
model_old.freeze()

model.train()  # IMPORTANT!

print("✅ LoRA applied")

def extract_score(text: str) -> float:
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

def pad_sequences(sequences, pad_token_id):
    if not sequences:
        return mx.array([])
    max_len = max(len(seq) for seq in sequences)
    padded = []
    for seq in sequences:
        if len(seq) < max_len:
            padding = mx.array([pad_token_id] * (max_len - len(seq)))
            padded.append(mx.concatenate([seq, padding]))
        else:
            padded.append(seq)
    return mx.stack(padded)

def calculate_log_probs(model, sequences, a_toks):
    logits = model(sequences)
    log_probs_full = nn.log_softmax(logits, axis=-1)
    batch_size, seq_len = sequences.shape
    _, ans_len = a_toks.shape
    start_pos = seq_len - ans_len
    answer_log_probs = log_probs_full[:, start_pos:start_pos+ans_len, :]
    indices = a_toks[:, :, None]
    selected_log_probs = mx.take_along_axis(answer_log_probs, indices, axis=-1).squeeze(-1)
    return mx.sum(selected_log_probs, axis=-1)

def grpo_loss_fn(model, model_ref, sequences, a_toks, advantages, old_log_probs, beta, epsilon):
    log_probs = calculate_log_probs(model, sequences, a_toks)
    log_probs_ref = calculate_log_probs(model_ref, sequences, a_toks)

    ratio = mx.exp(log_probs - old_log_probs)
    clipped_ratio = mx.clip(ratio, 1.0 - epsilon, 1.0 + epsilon)
    policy_reward = mx.minimum(ratio * advantages, clipped_ratio * advantages)

    log_ratio_for_kl = log_probs_ref - log_probs
    ratio_for_kl = mx.exp(log_ratio_for_kl)
    kl_div = ratio_for_kl - log_ratio_for_kl - 1

    loss = -mx.mean(policy_reward - beta * kl_div)
    return loss, mx.mean(policy_reward), mx.mean(kl_div)

# Test single iteration
print("\n3. Simulating single training iteration...")

batch_size = 2
group_size = 2  # Small for testing
max_ans_len = 5

rollout_sequences = []
rollout_rewards = []
rollout_a_toks = []

print("\n4. Generating rollouts...")
for i, example in enumerate(test_data):
    prompt = create_prompt(example)
    prompt_tokens = tokenizer.encode(prompt)
    print(f"\nExample {i+1}: '{example.text}' (target: {example.score})")
    print(f"  Prompt tokens: {len(prompt_tokens)}")

    group_rewards = []

    for j in range(group_size):
        # Generate
        response = generate(model_old, tokenizer, prompt=prompt, max_tokens=max_ans_len, verbose=False)
        print(f"  Generation {j+1}: '{response.strip()}'")

        # Extract answer tokens (response is ONLY the answer, not prompt+answer)
        answer_tokens = tokenizer.encode(response, add_special_tokens=False)
        print(f"    Answer tokens ({len(answer_tokens)}): {answer_tokens}")

        # Compute reward
        predicted = extract_score(response)
        error = abs(predicted - example.score)
        reward = 1.0 - error
        print(f"    Predicted: {predicted:.2f}, Error: {error:.3f}, Reward: {reward:.3f}")
        group_rewards.append(reward)

        # Store for optimization
        full_sequence = mx.array(prompt_tokens + answer_tokens)
        print(f"    Full sequence length: {len(full_sequence)}")
        rollout_sequences.append(full_sequence)
        rollout_a_toks.append(mx.array(answer_tokens))

    rollout_rewards.append(mx.array(group_rewards))

# Compute advantages
print("\n5. Computing advantages...")
advantages = []
for i, rewards in enumerate(rollout_rewards):
    mean_reward = mx.mean(rewards)
    std_reward = mx.sqrt(mx.var(rewards)) + 1e-8
    adv = (rewards - mean_reward) / std_reward
    print(f"  Group {i+1}: rewards={rewards}, mean={mean_reward.item():.3f}, std={std_reward.item():.3f}")
    print(f"           advantages={adv}")
    advantages.append(adv)

advantages = mx.concatenate(advantages)
print(f"\nAll advantages: {advantages}")

# Pad sequences
print("\n6. Padding sequences...")
sequences = pad_sequences(rollout_sequences, tokenizer.eos_token_id)
a_toks = pad_sequences(rollout_a_toks, tokenizer.eos_token_id)
print(f"  Sequences shape: {sequences.shape}")
print(f"  Answer tokens shape: {a_toks.shape}")

# Calculate old log probs
print("\n7. Calculating old log probs...")
old_log_probs = calculate_log_probs(model_old, sequences, a_toks)
print(f"  Old log probs: {old_log_probs}")

# Create gradient function and compute
print("\n8. Computing loss and gradients...")
loss_and_grad_fn = nn.value_and_grad(model, grpo_loss_fn)

beta = 0.01
epsilon = 0.2
(loss, policy_reward, kl_div), grads = loss_and_grad_fn(
    model, model_ref, sequences, a_toks, advantages, old_log_probs, beta, epsilon
)

print(f"  Loss: {loss.item():.6f}")
print(f"  Policy reward: {policy_reward.item():.6f}")
print(f"  KL div: {kl_div.item():.6f}")

# Check gradients
print("\n9. Checking gradients...")
flat_grads = tree_flatten(grads)
print(f"  Number of gradients: {len(flat_grads)}")

if len(flat_grads) == 0:
    print("  ❌ NO GRADIENTS!")
else:
    grad_norms = []
    for i, (path, grad) in enumerate(flat_grads):
        norm = float(mx.linalg.norm(grad).item())
        grad_norms.append(norm)
        if i < 5:
            print(f"    {path}: norm={norm:.6f}")

    print(f"\n  Gradient statistics:")
    print(f"    Max norm: {max(grad_norms):.6f}")
    print(f"    Min norm: {min(grad_norms):.6f}")
    print(f"    Mean norm: {np.mean(grad_norms):.6f}")

    if max(grad_norms) < 1e-6:
        print("  ❌ WARNING: Gradients are extremely small!")
    else:
        print("  ✅ Gradients look reasonable")

# Apply update
print("\n10. Applying optimizer update...")
optimizer = optim.Adam(learning_rate=1e-5)
optimizer.update(model, grads)
mx.eval(model.parameters(), optimizer.state)
print("  ✅ Parameters updated")

# Check if parameters changed
print("\n11. Verifying parameter changes...")
new_log_probs = calculate_log_probs(model, sequences, a_toks)
log_prob_diff = mx.abs(new_log_probs - old_log_probs)
print(f"  Old log probs: {old_log_probs}")
print(f"  New log probs: {new_log_probs}")
print(f"  Absolute difference: {log_prob_diff}")

if mx.max(log_prob_diff).item() < 1e-6:
    print("  ❌ Parameters didn't change significantly!")
else:
    print("  ✅ Parameters changed")

print("\n" + "="*70)
print("PIPELINE TEST COMPLETE")
print("="*70)
