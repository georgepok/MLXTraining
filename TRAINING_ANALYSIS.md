# Training Analysis & Issues

## Baseline Performance (Untrained Model)

- **Success Rate:** 53.8% (7/13 successful executions)
- **Exact Match Rate:** 46.2% (6/13 exact matches)
- **Average Reward:** +0.623

### Performance by Category:
| Category | Success | Exact Match | Avg Reward |
|----------|---------|-------------|------------|
| Aggregate | 4/4 (100%) | 3/4 (75%) | +1.375 |
| Basic | 2/3 (67%) | 2/3 (67%) | +0.900 |
| Complex | 1/3 (33%) | 1/3 (33%) | +0.267 |
| WHERE clauses | 0/3 (0%) | 0/3 (0%) | -0.300 |

## Previous Training Attempt Results

From notebook training (42 steps):
- **Final Reward:** -0.500
- **Success Rate:** 0.0%
- **Result:** Complete failure

## Root Cause Analysis

### Issue #1: Broken SQL Extraction (FIXED)
**Problem:** SQL extraction was skipping the actual SQL by removing `len(prompt)` characters from output that didn't include the prompt.

```python
# BEFORE (broken):
generated = output[len(prompt):].strip()  # Removed 50+ chars, skipping SQL!

# AFTER (fixed):
generated = output.strip()
if 'SQL:' in generated:
    generated = generated.split('SQL:', 1)[1].strip()
```

**Impact:** Training saw 0% success because ALL generated SQL was corrupted.

**Status:** ✅ FIXED in updated notebook

### Issue #2: Fake Gradients (MAJOR PROBLEM)
**Problem:** Gradients are random noise, not computed from actual model behavior.

```python
# Current (wrong):
grad = mx.random.normal(param.shape) * grad_scale
```

**Impact:**
- LoRA parameters update randomly
- No connection between SQL quality and parameter updates
- Model cannot learn from rewards
- This is **simulation**, not training

**Status:** ❌ NOT FIXED - fundamental issue

### Issue #3: No Real Backpropagation
**Problem:** The training loop doesn't compute gradients through the model's generation process.

**What's needed:**
- Compute log probabilities of generated tokens
- Calculate policy gradient: `∇ = reward * ∇log P(SQL | question)`
- Update LoRA parameters in direction that increases reward

**What we're doing:**
- Generating SQL
- Getting reward
- Updating parameters randomly (scaled by reward)

**Status:** ❌ NOT FIXED - requires major rewrite

### Issue #4: Small Training Scale
**Current:**
- 58 training examples
- 42 total steps (14 steps/epoch × 3 epochs)
- Batch size 4

**For real learning:**
- Need 500-1000+ steps minimum
- More diverse training data
- Proper learning rate scheduling

**Status:** ⚠️ PARTIALLY ADDRESSED (can run more steps, but won't help without real gradients)

## Why Training "Worked" But Didn't Learn

The training loop executed without errors after fixing SQL extraction, BUT:

1. **Random gradients** mean updates are essentially random walk
2. Even with correct SQL extraction and rewards, the parameter updates have no causal relationship to improving SQL generation
3. The model parameters change, but not in a way that makes it generate better SQL

## What Real Training Requires

### Option A: Full Policy Gradient Implementation (Hard)
```python
def compute_real_gradients(question, model, tokenizer, reward):
    # 1. Generate SQL while tracking token probabilities
    prompt_tokens = tokenizer.encode(prompt)
    logits_list = []

    # 2. Forward pass through model with cache
    for token in generated_tokens:
        logits = model(...)
        logits_list.append(logits)

    # 3. Compute log probabilities
    log_probs = compute_log_probs(logits_list, generated_tokens)

    # 4. Policy gradient: grad = reward * grad(log_prob)
    loss = -reward * sum(log_probs)

    # 5. Backprop through model (only LoRA params)
    grads = mx.grad(loss, lora_params)

    return grads
```

**Challenges:**
- Need to modify mlx_lm's generate() to return logits
- Complex gradient computation
- Memory intensive

### Option B: Supervised Fine-Tuning (Easier, Better)
```python
def compute_supervised_loss(question, correct_sql, model, tokenizer, lora_params):
    # 1. Encode input and target
    input_ids = tokenizer.encode(f"Generate SQL for: {question}\nSQL:")
    target_ids = tokenizer.encode(correct_sql)

    # 2. Forward pass
    logits = model(input_ids)

    # 3. Cross-entropy loss between logits and target tokens
    loss = cross_entropy(logits, target_ids)

    # 4. Backprop (only through LoRA)
    grads = mx.grad(loss, lora_params)

    return loss, grads
```

**Advantages:**
- Standard supervised learning
- Proven to work
- Simpler implementation
- Better sample efficiency

### Option C: Use MLX's Built-in LoRA Training
MLX likely has built-in LoRA training utilities that handle gradients correctly.

## Recommendation

**Immediate Fix: Switch to Supervised Fine-Tuning**

1. Use the correct SQL examples we already have
2. Implement proper cross-entropy loss
3. Use mx.grad() for real gradients
4. This will actually teach the model to generate correct SQL

**Why this is better:**
- Proven approach (standard fine-tuning)
- Simpler than policy gradients
- Will actually learn
- Can validate improvement easily

## Expected Results After Real Training

With proper supervised fine-tuning:

| Metric | Before | After (Target) |
|--------|--------|----------------|
| Success Rate | 53.8% | 75-85% |
| WHERE clauses | 0% | 60-80% |
| Complex queries | 33% | 65-75% |
| Avg Reward | +0.623 | +0.8 to +1.0 |

The model should learn:
- Correct table/column names
- Proper SQL syntax patterns
- When to use different clauses
- Better matching to expected outputs

## Next Steps

1. ✅ Identified all training issues
2. 🔄 Implement supervised fine-tuning with real gradients
3. 🔄 Run training for 500+ steps
4. 🔄 Validate improvement on test set
5. 🔄 Compare before/after metrics
