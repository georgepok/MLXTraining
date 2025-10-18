# GRPO Reference Implementation Analysis

## Source
**Notebook:** https://github.com/searlion/mlx-finetuning/blob/main/MLX%20LM%20GRPO.ipynb
**Author:** searlion
**Task:** HellaSwag (common sense reasoning)
**Model:** Phi-3-mini-4k-instruct (3.8B params)

---

## Key Breakthrough: How to Get True Policy Gradients in MLX

### The Problem We Had
Our implementation couldn't compute `∇log π(a|s)` because we thought we needed to extract log probabilities from inside `mlx_lm.generate()`.

### The Solution
**Don't try to extract from generate() - recompute everything!**

```python
# Step 1: Use generate() to get the text (no log probs needed here)
response = generate(model_old, tokenizer, prompt_tokens, max_tokens=max_ans_len)
answer_tokens = tokenizer.encode(response, add_special_tokens=False)

# Step 2: Reconstruct the full sequence
full_sequence = mx.array(prompt_tokens + answer_tokens)

# Step 3: Recompute log probabilities by running model on full sequence
def calculate_log_probs(model, sequences, a_toks):
    # Forward pass through model
    logits = model(sequences)  # ← This gives us logits!

    # Convert to log probabilities
    log_probs_full = nn.log_softmax(logits, axis=-1)

    # Extract answer portion
    batch_size, seq_len = sequences.shape
    _, ans_len = a_toks.shape
    start_pos = seq_len - ans_len
    answer_log_probs = log_probs_full[:, start_pos:start_pos+ans_len, :]

    # Gather log probs for actual tokens that were generated
    indices = a_toks[:, :, None]
    selected_log_probs = mx.take_along_axis(answer_log_probs, indices, axis=-1).squeeze(-1)

    # Sum across sequence
    return mx.sum(selected_log_probs, axis=-1)
```

**Key Insight:** `model(sequences)` gives us logits for every position! We can:
1. Convert to log probabilities with `log_softmax`
2. Extract the positions where answers were generated
3. Gather the log probabilities for the actual tokens that were sampled
4. Get a differentiable path from parameters → log probs → loss

---

## GRPO Loss Function

```python
def grpo_loss_fn(model, model_ref, sequences, a_toks, advantages, old_log_probs, beta, epsilon):
    # 1. Get log probs from trainable model (π_θ)
    log_probs = calculate_log_probs(model, sequences, a_toks)

    # 2. Get log probs from reference model (π_ref) for KL penalty
    log_probs_ref = calculate_log_probs(model_ref, sequences, a_toks)

    # 3. PPO-clip objective
    ratio = mx.exp(log_probs - old_log_probs)
    clipped_ratio = mx.clip(ratio, 1.0 - epsilon, 1.0 + epsilon)
    policy_reward = mx.minimum(ratio * advantages, clipped_ratio * advantages)

    # 4. KL penalty: r - log(r) - 1
    # where r = π_ref / π_θ
    log_ratio_for_kl = log_probs_ref - log_probs
    ratio_for_kl = mx.exp(log_ratio_for_kl)
    kl_div = ratio_for_kl - log_ratio_for_kl - 1

    # 5. Combined loss (maximize policy reward, minimize KL)
    loss = -mx.mean(policy_reward - beta * kl_div)
    return loss, mx.mean(policy_reward), mx.mean(kl_div)
```

**Components:**
- **Policy Reward:** PPO-clip objective with advantages
- **KL Penalty:** Prevents policy from diverging from reference
- **Returns:** Loss for optimization + metrics for tracking

---

## Training Loop Structure

```python
def grpo_train_loop(model, model_old, model_ref, tokenizer, optimizer, train_set,
                    iters=200, group_size=4, batch_size=2, epsilon=0.2, beta=0.01,
                    update_every=10, max_ans_len=4):

    # Create gradient function using value_and_grad
    loss_and_grad_fn = nn.value_and_grad(model, grpo_loss_fn)

    for it in range(iters):
        # 1. Sample batch
        batch_prompts = [...]
        batch_answers = [...]

        # 2. Rollout: Generate G responses per prompt using model_old
        rollout_sequences = []
        rollout_rewards = []
        rollout_a_toks = []

        for i in range(batch_size):
            for _ in range(group_size):
                # Generate response
                response = generate(model_old, tokenizer, prompt_tokens, max_tokens=max_ans_len)
                answer_tokens = tokenizer.encode(response, add_special_tokens=False)

                # Compute reward
                reward = compute_reward(response, ground_truth)

                # Store
                full_sequence = mx.array(prompt_tokens + answer_tokens)
                rollout_sequences.append(full_sequence)
                rollout_a_toks.append(mx.array(answer_tokens))

        # 3. Compute group-relative advantages
        advantages = []
        for rewards in rollout_rewards:  # Each group
            mean_reward = mx.mean(rewards)
            std_reward = mx.sqrt(mx.var(rewards)) + 1e-8
            adv = (rewards - mean_reward) / std_reward  # Normalized!
            advantages.append(adv)

        # 4. Pad sequences to same length
        sequences = pad_sequences(rollout_sequences, tokenizer.pad_token_id)
        a_toks = pad_sequences(rollout_a_toks, tokenizer.pad_token_id)

        # 5. Calculate old log probs (for PPO ratio)
        old_log_probs = calculate_log_probs(model_old, sequences, a_toks)

        # 6. Optimization step with TRUE GRADIENTS
        (loss, policy_reward, kl_div), grads = loss_and_grad_fn(
            model, model_ref, sequences, a_toks, advantages, old_log_probs, beta, epsilon
        )

        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        # 7. Sync model_old periodically
        if (it + 1) % update_every == 0:
            model_old.update(model.parameters())
```

---

## Three Model Architecture

This is crucial for GRPO:

### 1. `model` (π_θ) - Training Model
- Has LoRA adapters attached
- Only LoRA parameters are trainable
- Updated every iteration via gradients
- This is the policy we're optimizing

### 2. `model_old` (π_θ_old) - Rollout Model
- Copy of `model` with same LoRA structure
- **Frozen** during optimization
- Used for generating rollouts (responses)
- Synced with `model` every N iterations
- Why? Stabilizes training - generate with slightly stale policy

### 3. `model_ref` (π_ref) - Reference Model
- Original pretrained model
- **Completely frozen** (no LoRA, never updated)
- Used for KL divergence penalty
- Why? Prevents policy from straying too far from original behavior

```python
# Load models
model, tokenizer = load(model_path)
model_ref, _ = load(model_path)
model_ref.freeze()

# Freeze base and add LoRA to main model
model.freeze()
linear_to_lora_layers(model, num_layers, lora_config)

# Create old model with same structure
model_old, _ = load(model_path)
linear_to_lora_layers(model_old, num_layers, lora_config)
model_old.update(model.parameters())  # Sync weights
model_old.freeze()
```

---

## Why This Works: The Magic of mx.take_along_axis

The key to extracting log probabilities for generated tokens:

```python
# logits shape: (batch, seq_len, vocab_size)
# We want: log_prob of token that was actually generated at each position

log_probs_full = nn.log_softmax(logits, axis=-1)
# Now: (batch, seq_len, vocab_size) with log probabilities

# a_toks contains the actual token IDs that were generated
# Shape: (batch, ans_len)

# Reshape for gathering: (batch, ans_len, 1)
indices = a_toks[:, :, None]

# Extract only the answer portion
answer_log_probs = log_probs_full[:, start_pos:start_pos+ans_len, :]

# Gather: For each position, get log_prob of the actual token
selected_log_probs = mx.take_along_axis(answer_log_probs, indices, axis=-1)
# Shape: (batch, ans_len, 1)

# Squeeze and sum
selected_log_probs = selected_log_probs.squeeze(-1)
# Shape: (batch, ans_len)

return mx.sum(selected_log_probs, axis=-1)
# Shape: (batch,) - total log prob for each sequence
```

**What this gives us:**
- For each generated sequence, we have `log π(a|s)`
- This is fully differentiable w.r.t. model parameters!
- We can compute `∇_θ log π(a|s)` via `mx.grad()`

---

## Advantage Computation (The "Group Relative" Part)

```python
# For each prompt, we generated G responses (a group)
for rewards in rollout_rewards:  # Each group has G rewards
    mean_reward = mx.mean(rewards)
    std_reward = mx.sqrt(mx.var(rewards)) + 1e-8

    # Normalize advantages within group
    adv = (rewards - mean_reward) / std_reward
    advantages.append(adv)
```

**Why normalize?**
- Mean centering: `rewards - mean_reward` makes advantages zero-sum within group
- Standard deviation: Normalizing by std makes advantages comparable across groups
- Stability: Prevents extreme advantage values from destabilizing training

**Comparison:**
```
Standard RL:     A = R - baseline
Group Relative:  A = (R - mean(R_group)) / std(R_group)

Benefits:
- Lower variance (within-group comparison)
- Automatic normalization across different reward scales
- More stable training
```

---

## Using nn.value_and_grad for True Gradients

This is the crucial piece that enables true policy gradients:

```python
# Create gradient function
loss_and_grad_fn = nn.value_and_grad(model, grpo_loss_fn)

# Call it to get both loss value and gradients
(loss, policy_reward, kl_div), grads = loss_and_grad_fn(
    model, model_ref, sequences, a_toks, advantages, old_log_probs, beta, epsilon
)

# Update parameters
optimizer.update(model, grads)
```

**What `nn.value_and_grad` does:**
1. Calls `grpo_loss_fn` with provided arguments
2. Computes gradients of loss w.r.t. model's trainable parameters
3. Returns both the function output AND the gradients

**The gradient path:**
```
Parameters (LoRA weights)
  ↓
Model forward pass: logits = model(sequences)
  ↓
Log probabilities: log_probs = log_softmax(logits)[actual_tokens]
  ↓
Ratio: ratio = exp(log_probs - old_log_probs)
  ↓
Policy reward: min(ratio * adv, clip(ratio) * adv)
  ↓
Loss: -mean(policy_reward - beta * kl_div)
  ↓
∇Loss w.r.t. LoRA parameters ← This is what we get!
```

This is **REAL backpropagation**, not random noise!

---

## Differences from Our Failed Implementation

### Our Implementation (grpo_sentiment_training.py)

```python
# ❌ Random gradients
for name, param in lora_params_dict.items():
    grad_direction = mx.random.normal(param.shape)  # Random!
    grad = grad_direction * grad_scale
    gradients[name] = grad
```

**Result:** 0% improvement

### Reference Implementation

```python
# ✅ True policy gradients via automatic differentiation
loss_and_grad_fn = nn.value_and_grad(model, grpo_loss_fn)
(loss, ...), grads = loss_and_grad_fn(model, model_ref, sequences, ...)
optimizer.update(model, grads)
```

**Result:** Actual learning

---

## Hyperparameters in Reference

```python
learning_rate = 1e-5      # Very small for stability
iters = 200               # Training iterations
group_size = 4            # G in paper - completions per prompt
batch_size = 4            # Prompts per iteration
epsilon = 0.2             # PPO clip parameter (common default)
beta = 0.02               # KL penalty coefficient
update_every = 10         # Sync model_old every N iters
max_ans_len = 4           # Max tokens to generate
```

**LoRA Config:**
```python
lora_config = {
    "num_layers": 8,       # Number of layers to apply LoRA
    "lora_parameters": {
        "rank": 8,
        "scale": 10.0,
        "dropout": 0.0,
    }
}
```

---

## Adaptation Strategy for Our Sentiment Task

### Changes Needed:

1. **Reward Function:**
```python
# Reference (HellaSwag):
reward = 1.0 if batch_answers[i] in response else 0.0

# Our sentiment task:
def compute_reward(response, target_score):
    predicted = extract_score(response)
    error = abs(predicted - target_score)
    reward = 1.0 - error  # Higher reward for lower error
    return reward
```

2. **Prompt Format:**
```python
# Reference uses chat template:
messages = [
    {"role": "user", "content": instruction},
    {"role": "assistant", "content": prompt_text}
]

# Our sentiment task:
def create_prompt(example):
    return f"""Rate this review's sentiment from 0.0 to 1.0.
Review: "{example.text}"
Rating:"""
```

3. **Answer Length:**
```python
# Reference: max_ans_len = 4 (for "ending1", "ending2", etc.)
# Ours: max_ans_len = 5 (for "0.85" or "0.923")
```

4. **Model Size:**
```python
# Reference: Phi-3-mini-4k-instruct (3.8B params)
# Ours: Qwen2.5-0.5B-Instruct-4bit (0.5B params)
# May need to adjust learning rate / LoRA config for smaller model
```

---

## Implementation Checklist

To adapt this to our sentiment task:

- [ ] Load three models (model, model_old, model_ref)
- [ ] Apply LoRA to model and model_old, freeze model_ref
- [ ] Implement `calculate_log_probs()` function
- [ ] Implement `grpo_loss_fn()` with PPO-clip and KL penalty
- [ ] Implement `pad_sequences()` helper
- [ ] Create sentiment-specific reward function
- [ ] Implement `grpo_train_loop()` with rollout generation
- [ ] Use `nn.value_and_grad()` for true gradients
- [ ] Evaluate on validation set

---

## Expected Results

Based on the reference implementation:
- **Training should show:** Loss decreasing, mean reward increasing
- **Validation should show:** Improved sentiment prediction accuracy
- **Compared to our working demo:** Should see similar or better results (42-66% improvement)

**Why this should work:**
1. ✅ True policy gradients via `nn.value_and_grad()`
2. ✅ Proper log probability calculation via `model(sequences)`
3. ✅ PPO-clip objective for stable updates
4. ✅ KL penalty to prevent divergence
5. ✅ Group-relative advantages for variance reduction

---

## Memory and Efficiency Notes

**Memory usage:** Higher than standard training because:
- Three models in memory (though model_ref can be quantized)
- Storing rollout sequences and log probs
- Larger effective batch size (batch_size × group_size completions)

**Tips from reference:**
- Use smaller batch_size or group_size if OOM
- Reference uses batch_size=4, group_size=4 → 16 completions per iteration
- Our smaller model may allow larger batches
- Quantize model_ref (it's never trained, only used for forward passes)

---

## Conclusion

This reference implementation shows that **true GRPO with real policy gradients IS possible in MLX**!

The key insights:
1. Don't try to extract internals from `generate()` - just use it for sampling
2. Recompute log probabilities by running `model(sequences)` to get logits
3. Use `nn.value_and_grad()` for automatic differentiation
4. Maintain three models for stable policy optimization

This is exactly what we need to implement a working GRPO demonstration.
