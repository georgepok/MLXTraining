#!/usr/bin/env python3
"""
Test if LoRA adapters are actually used during generation
"""

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.tuner import linear_to_lora_layers
from mlx.utils import tree_flatten
import mlx.optimizers as optim
import mlx.nn as nn

print("="*70)
print("TEST: Do LoRA adapters affect generation?")
print("="*70)

# Load model
print("\n1. Loading model...")
model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
print("   ✅ Loaded")

# Test prompt
test_prompt = """Rate this review's sentiment from 0.0 (very negative) to 1.0 (very positive).
Reply with just the number.

Review: "This was okay, nothing special."

Rating:"""

# Generate BEFORE LoRA
print("\n2. Generation BEFORE applying LoRA (base model)...")
response_before = generate(model, tokenizer, prompt=test_prompt, max_tokens=5, verbose=False)
print(f"   Response: '{response_before}'")

# Apply LoRA
print("\n3. Applying LoRA...")
lora_config = {
    "num_layers": 12,
    "lora_parameters": {"rank": 8, "scale": 16.0, "dropout": 0.0}
}

model.freeze()
linear_to_lora_layers(model, lora_config["num_layers"], lora_config["lora_parameters"])
print("   ✅ LoRA applied (but not trained yet)")

# Generate AFTER LoRA (untrained)
print("\n4. Generation AFTER applying LoRA (untrained adapters)...")
response_after_lora = generate(model, tokenizer, prompt=test_prompt, max_tokens=5, verbose=False)
print(f"   Response: '{response_after_lora}'")

if response_before == response_after_lora:
    print("   ✅ Same as before (expected - adapters are initialized to identity)")
else:
    print("   ⚠️  Different! (unexpected - untrained LoRA shouldn't change output)")

# Train LoRA a bit
print("\n5. Training LoRA adapters for 100 steps...")

# Simple loss to change the model
def loss_fn(model, sequence, a_toks):
    logits = model(sequence)
    log_probs = nn.log_softmax(logits, axis=-1)

    batch_size, seq_len = sequence.shape
    _, ans_len = a_toks.shape
    start_pos = seq_len - ans_len

    answer_log_probs = log_probs[:, start_pos:start_pos+ans_len, :]
    indices = a_toks[:, :, None]
    selected = mx.take_along_axis(answer_log_probs, indices, axis=-1).squeeze(-1)

    return -mx.mean(selected)

prompt_tokens = tokenizer.encode(test_prompt)
answer_tokens = tokenizer.encode("0.3", add_special_tokens=False)  # Train to predict 0.3
sequence = mx.array(prompt_tokens + answer_tokens)[None, :]
a_toks = mx.array(answer_tokens)[None, :]

optimizer = optim.Adam(learning_rate=1e-5)  # Higher LR for visible changes
loss_and_grad = nn.value_and_grad(model, loss_fn)

for step in range(100):
    loss, grads = loss_and_grad(model, sequence, a_toks)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

    if step % 20 == 0:
        print(f"   Step {step}: Loss = {loss.item():.4f}")

print(f"   Final loss: {loss.item():.4f}")

# Generate AFTER training
print("\n6. Generation AFTER training LoRA...")
response_trained = generate(model, tokenizer, prompt=test_prompt, max_tokens=5, verbose=False)
print(f"   Response: '{response_trained}'")

if response_trained == response_before:
    print("\n   ❌ CRITICAL: Response UNCHANGED after training!")
    print("   ❌ This means LoRA adapters are NOT being used in generation!")
    print("\n   POSSIBLE CAUSES:")
    print("   • generate() might not use LoRA layers")
    print("   • model.eval() might disable LoRA")
    print("   • LoRA implementation issue")
elif response_trained == response_after_lora:
    print("\n   ❌ Response same as untrained LoRA")
    print("   ❌ Training didn't affect the model")
else:
    print("\n   ✅ Response CHANGED after training!")
    print("   ✅ LoRA adapters ARE being used correctly")

# Test model.eval() mode
print("\n7. Testing model.eval() mode...")
model.eval()
response_eval = generate(model, tokenizer, prompt=test_prompt, max_tokens=5, verbose=False)
model.train()
print(f"   Response in eval mode: '{response_eval}'")

if response_eval == response_trained:
    print("   ✅ eval() mode still uses trained LoRA")
elif response_eval == response_before:
    print("   ❌ CRITICAL: eval() mode reverts to base model!")
    print("   ❌ This explains why validation shows no improvement!")
else:
    print("   ⚠️  eval() gives different output (unexpected)")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

if response_trained == response_before:
    print("\n❌ LORA NOT WORKING IN GENERATION")
    print("   This is why GRPO shows 0% improvement!")
elif response_eval == response_before:
    print("\n❌ eval() MODE DISABLES LORA")
    print("   This is why validation shows no improvement!")
    print("\n   FIX: Don't use model.eval() for validation")
else:
    print("\n✅ LoRA working correctly")
    print("   Issue must be elsewhere")

print("="*70)
