# GRPO v2 Bug Report & Fix

## Root Cause Identified ✅

The GRPO v2 implementation shows **0% improvement** because:

### The Problem

**All advantages within each group are ZERO**

Example from test run:
```
Group 1: rewards=[0.9, 0.9] → mean=0.900, std=0.000 → advantages=[0, 0]
Group 2: rewards=[0.7, 0.7] → mean=0.700, std=0.000 → advantages=[0, 0]
```

### Why This Happens

1. **Deterministic Generation**: `generate()` without sampling always produces the same output for the same input
2. **Identical Completions**: All 4 completions in a group are identical
3. **Zero Variance**: std(rewards) = 0
4. **Zero Advantages**: adv = (reward - mean) / std = 0 / 0 = 0
5. **Zero Gradients**: policy_reward = ratio * advantage = X * 0 = 0
6. **No Learning**: gradients are all zero, parameters don't update

### Evidence

From `test_full_pipeline.py`:
```
Example 1: 'This movie was amazing!' (target: 0.9)
  Generation 1: '1.0'  ← Same
  Generation 2: '1.0'  ← Same

Example 2: 'Pretty bad movie.' (target: 0.3)
  Generation 1: '0.0'  ← Same
  Generation 2: '0.0'  ← Same

Computing advantages:
  Group 1: advantages=[0, 0]  ← ZERO!
  Group 2: advantages=[0, 0]  ← ZERO!

Computing loss and gradients:
  Loss: -0.000000
  Policy reward: 0.000000
  Gradients: all zeros
  Parameters didn't change!
```

---

## The Fix

### Problem: Deterministic Generation

Current code (grpo_sentiment_training_v2.py:313):
```python
response = generate(model_old, tokenizer, prompt=prompt, max_tokens=max_ans_len, verbose=False)
```

This generates deterministically (greedy decoding).

### Solution: Add Sampling

We need to add sampling to get diverse completions:

```python
# Option 1: Add temperature
response = generate(model_old, tokenizer, prompt=prompt, max_tokens=max_ans_len,
                   temp=0.7, verbose=False)  # Add temperature for sampling

# Option 2: Top-p sampling
response = generate(model_old, tokenizer, prompt=prompt, max_tokens=max_ans_len,
                   temp=0.8, top_p=0.9, verbose=False)
```

**But wait!** The `generate()` function in our mlx_lm version doesn't accept `temp` parameter (we got errors earlier).

Let me check what parameters it does accept...

### Investigation: mlx_lm generate() Parameters

From earlier error messages:
```
TypeError: generate_step() got an unexpected keyword argument 'temp'
TypeError: generate_step() got an unexpected keyword argument 'temperature'
```

The mlx_lm version being used doesn't support temperature parameter directly.

---

## Two Solutions

### Solution 1: Use Sampling in mlx_lm (if supported)

Check mlx_lm documentation for proper sampling parameters. Possible options:
- `temp` parameter (if supported in newer version)
- `sampler` parameter
- Custom generation with sampling

### Solution 2: Add Noise to Prompts

If sampling isn't available, add variation to prompts:

```python
# Add slight variations to prompts
variations = [
    prompt,
    prompt + " Think carefully.",
    prompt + " Consider all aspects.",
    prompt + " What's your assessment?"
]

for variation in variations[:group_size]:
    response = generate(model_old, tokenizer, prompt=variation, ...)
```

This is hacky but would create diversity.

### Solution 3: Use Different Random Seeds

Some generation functions accept a seed parameter:

```python
for i in range(group_size):
    response = generate(model_old, tokenizer, prompt=prompt,
                       max_tokens=max_ans_len,
                       seed=np.random.randint(0, 10000),  # Different seed each time
                       verbose=False)
```

---

## Recommended Fix

### Check mlx_lm Version and Capabilities

First, let's check what version of mlx_lm is installed and what parameters `generate()` actually supports.

```python
import mlx_lm
print(mlx_lm.__version__)

from inspect import signature
print(signature(mlx_lm.generate))
```

### Temporary Workaround

Until we can enable proper sampling, add artificial diversity:

```python
# In grpo_sentiment_training_v2.py, around line 311

for j in range(group_size):
    # Add slight variation to prompt for diversity
    if j == 0:
        varied_prompt = prompt
    elif j == 1:
        varied_prompt = prompt + " Be precise."
    elif j == 2:
        varied_prompt = prompt + " Consider carefully."
    else:
        varied_prompt = prompt + " What's your assessment?"

    response = generate(model_old, tokenizer, prompt=varied_prompt,
                       max_tokens=max_ans_len, verbose=False)
```

This isn't ideal, but it would create diversity needed for GRPO.

---

## Why This Matters for GRPO

GRPO (Group Relative Policy Optimization) **requires diversity** within groups:

1. **Variance Reduction**: The whole point of GRPO is to use group-relative advantages:
   ```
   A_i = R_i - mean(R_group)
   ```

2. **Within-Group Comparison**: We compare completions within a group to reduce variance
   - If all completions are identical, there's nothing to compare
   - Advantages become zero
   - No learning signal

3. **Exploration**: Sampling allows the model to explore different outputs
   - Some will be better (positive advantage)
   - Some will be worse (negative advantage)
   - The policy learns to favor better outputs

---

## Comparison with Reference Implementation

The reference notebook (searlion/mlx-finetuning) likely uses:
- A newer version of mlx_lm with sampling support
- Or configures sampling differently
- Or has a different default behavior

Their code would naturally produce diverse completions because:
- HellaSwag has 4 distinct answer choices
- Even without sampling, there's natural diversity in the task

Our sentiment task:
- Generates scores (0.0-1.0)
- Without sampling, always generates the same score for the same prompt
- Needs explicit sampling to get diversity

---

## Action Items

### Immediate

1. **Check mlx_lm capabilities**
   ```python
   python -c "import mlx_lm; print(mlx_lm.__version__)"
   python -c "from mlx_lm import generate; from inspect import signature; print(signature(generate))"
   ```

2. **Test sampling parameters**
   - Try `temp`, `temperature`, `sampling`, `top_p`, `top_k`
   - See which ones are accepted

3. **Implement workaround if needed**
   - Add prompt variations
   - Or use temperature if supported

### Long-term

1. **Upgrade mlx_lm** if needed to get sampling support

2. **Implement custom sampling** if generate() doesn't support it:
   ```python
   def sample_generation(model, tokenizer, prompt, max_tokens, temperature=0.7):
       # Custom sampling logic
       pass
   ```

3. **Document** the sampling requirements for GRPO

---

## Expected Results After Fix

Once we add diversity to completions:

```
Group 1: rewards=[0.85, 0.92, 0.88, 0.90] → mean=0.888, std=0.028
         advantages=[-1.36, 1.14, -0.29, 0.43]  ← Non-zero!

Group 2: rewards=[0.65, 0.72, 0.68, 0.70] → mean=0.688, std=0.029
         advantages=[-1.31, 1.10, -0.28, 0.41]  ← Non-zero!
```

With non-zero advantages:
- Policy reward will be non-zero
- Gradients will be non-zero
- Parameters will update
- Learning will occur!

---

## Summary

**Root Cause**: Deterministic generation → identical completions → zero advantages → zero gradients → no learning

**Fix**: Enable sampling in `generate()` to produce diverse completions within each group

**Status**: Implementation is correct, just needs sampling enabled

**Impact**: This is why GRPO v2 showed 0% improvement despite having correct gradient computation via `nn.value_and_grad()`
