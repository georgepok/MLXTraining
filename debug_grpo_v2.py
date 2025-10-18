#!/usr/bin/env python3
"""
Debug GRPO v2 - Investigate why learning doesn't occur
"""

import mlx.core as mx
import mlx.nn as nn
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

# Simple test example
test_example = SentimentExample("This movie was amazing!", 0.90, "very_positive")

print("="*70)
print("DEBUGGING GRPO V2")
print("="*70)

# Load model
print("\n1. Loading model...")
model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")

# Apply LoRA
print("\n2. Applying LoRA...")
lora_config = {
    "num_layers": 2,  # Just 2 layers for testing
    "lora_parameters": {
        "rank": 8,
        "scale": 16.0,
        "dropout": 0.0,
    }
}
model.freeze()
linear_to_lora_layers(model, lora_config["num_layers"], lora_config["lora_parameters"])
print("✅ LoRA applied")

def create_prompt(example: SentimentExample) -> str:
    return f"""Rate this review's sentiment from 0.0 (very negative) to 1.0 (very positive).
Reply with just the number.

Review: "{example.text}"

Rating:"""

def extract_score(text: str) -> float:
    match = re.search(r'(\d+\.?\d*)', text)
    if match:
        score = float(match.group(1))
        if score > 1.0:
            score /= 10.0
        return max(0.0, min(1.0, score))
    return 0.5

# Test generation
print("\n3. Testing generation...")
prompt = create_prompt(test_example)
print(f"Prompt:\n{prompt}")
print("\nPrompt length:", len(prompt))

# Encode prompt
prompt_tokens = tokenizer.encode(prompt)
print(f"\nPrompt tokens: {len(prompt_tokens)} tokens")
print(f"First 10: {prompt_tokens[:10]}")
print(f"Last 10: {prompt_tokens[-10:]}")

# Generate response
print("\n4. Generating response...")
response = generate(model, tokenizer, prompt=prompt, max_tokens=5, verbose=False)
print(f"Full response:\n'{response}'")
print(f"Response length: {len(response)}")

# Check if response contains prompt
print("\n5. Analyzing response structure...")
if prompt in response:
    print("✅ Response contains full prompt")
    answer_str = response[len(prompt):].strip()
    print(f"Answer (by string slicing): '{answer_str}'")
else:
    print("❌ Response doesn't contain full prompt")
    print("Trying to extract after 'Rating:'...")
    if 'Rating:' in response:
        answer_str = response.split('Rating:', 1)[1].strip()
        print(f"Answer (by split): '{answer_str}'")
    else:
        print("❌ No 'Rating:' marker found")
        answer_str = response[-20:]  # Last 20 chars
        print(f"Answer (last 20 chars): '{answer_str}'")

# Encode full response
print("\n6. Tokenizing response...")
response_tokens = tokenizer.encode(response, add_special_tokens=False)
print(f"Response tokens: {len(response_tokens)} tokens")
print(f"Response tokens: {response_tokens}")

# Try to extract answer tokens
print("\n7. Extracting answer tokens...")
print(f"Method 1: response_tokens[len(prompt_tokens):]")
if len(response_tokens) > len(prompt_tokens):
    answer_tokens_1 = response_tokens[len(prompt_tokens):]
    print(f"  Answer tokens: {answer_tokens_1}")
    print(f"  Decoded: '{tokenizer.decode(answer_tokens_1)}'")
else:
    print(f"  ❌ Response tokens ({len(response_tokens)}) <= prompt tokens ({len(prompt_tokens)})")
    answer_tokens_1 = []

print(f"\nMethod 2: Encode answer string directly")
answer_tokens_2 = tokenizer.encode(answer_str, add_special_tokens=False)
print(f"  Answer tokens: {answer_tokens_2}")
print(f"  Decoded: '{tokenizer.decode(answer_tokens_2)}'")

# Test log probability computation
print("\n8. Testing log probability computation...")

def calculate_log_probs(model, sequences, a_toks):
    """Calculate log probabilities for generated answer tokens."""
    print(f"  Input sequences shape: {sequences.shape}")
    print(f"  Input a_toks shape: {a_toks.shape}")

    # Forward pass
    logits = model(sequences)
    print(f"  Logits shape: {logits.shape}")

    # Convert to log probabilities
    log_probs_full = nn.log_softmax(logits, axis=-1)
    print(f"  Log probs full shape: {log_probs_full.shape}")

    # Find answer position
    batch_size, seq_len = sequences.shape
    _, ans_len = a_toks.shape
    start_pos = seq_len - ans_len
    print(f"  Start pos: {start_pos}, Answer length: {ans_len}")

    # Extract answer portion
    answer_log_probs = log_probs_full[:, start_pos:start_pos+ans_len, :]
    print(f"  Answer log probs shape: {answer_log_probs.shape}")

    # Gather log probs for actual tokens
    indices = a_toks[:, :, None]
    print(f"  Indices shape: {indices.shape}")

    selected_log_probs = mx.take_along_axis(answer_log_probs, indices, axis=-1).squeeze(-1)
    print(f"  Selected log probs shape: {selected_log_probs.shape}")
    print(f"  Selected log probs values: {selected_log_probs}")

    # Sum across sequence
    total_log_prob = mx.sum(selected_log_probs, axis=-1)
    print(f"  Total log prob shape: {total_log_prob.shape}")
    print(f"  Total log prob value: {total_log_prob}")

    return total_log_prob

# Create test sequences
if len(answer_tokens_2) > 0:
    print("\nUsing answer tokens from method 2 (encoded answer string)")
    full_sequence = mx.array([prompt_tokens + answer_tokens_2])
    a_toks = mx.array([answer_tokens_2])

    print(f"\nFull sequence shape: {full_sequence.shape}")
    print(f"Answer tokens shape: {a_toks.shape}")

    print("\nCalculating log probabilities...")
    log_prob = calculate_log_probs(model, full_sequence, a_toks)
    print(f"\n✅ Final log probability: {log_prob.item():.6f}")
else:
    print("\n❌ No answer tokens to test!")

# Test with dummy sequence
print("\n" + "="*70)
print("9. Testing with dummy sequence...")
dummy_seq = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
dummy_a_toks = mx.array([[6, 7, 8]])
print(f"Dummy sequence: {dummy_seq}")
print(f"Dummy answer tokens: {dummy_a_toks}")

dummy_log_prob = calculate_log_probs(model, dummy_seq, dummy_a_toks)
print(f"\n✅ Dummy log probability: {dummy_log_prob.item():.6f}")

# Test gradient computation
print("\n" + "="*70)
print("10. Testing gradient computation...")

model_ref, _ = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
model_ref.freeze()

def simple_loss_fn(model, model_ref, sequences, a_toks):
    """Simplified loss for testing"""
    log_probs = calculate_log_probs(model, sequences, a_toks)
    log_probs_ref = calculate_log_probs(model_ref, sequences, a_toks)

    # Simple loss: negative log probability
    loss = -mx.mean(log_probs)

    return loss

print("\nCreating gradient function...")
loss_and_grad_fn = nn.value_and_grad(model, simple_loss_fn)

print("\nComputing loss and gradients...")
if len(answer_tokens_2) > 0:
    loss, grads = loss_and_grad_fn(model, model_ref, full_sequence, a_toks)

    print(f"\n✅ Loss: {loss.item():.6f}")

    # Check gradient magnitudes
    grad_norms = []
    for name, grad in grads.items():
        if 'lora' in name:
            norm = float(mx.linalg.norm(grad).item())
            grad_norms.append(norm)
            if len(grad_norms) <= 5:  # Print first 5
                print(f"  {name}: grad norm = {norm:.6f}")

    print(f"\nGradient statistics:")
    print(f"  Number of LoRA gradients: {len(grad_norms)}")
    print(f"  Max gradient norm: {max(grad_norms) if grad_norms else 0:.6f}")
    print(f"  Min gradient norm: {min(grad_norms) if grad_norms else 0:.6f}")
    print(f"  Mean gradient norm: {np.mean(grad_norms) if grad_norms else 0:.6f}")

    if max(grad_norms) < 1e-6:
        print("\n❌ WARNING: Gradients are extremely small!")
    else:
        print("\n✅ Gradients look reasonable")
else:
    print("\n❌ Cannot test gradients without answer tokens!")

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)
