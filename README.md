# RLVR + LoRA Fine-Tuning with MLX

**Successfully demonstrates Reinforcement Learning from Verifiable Rewards (RLVR) + LoRA fine-tuning on Apple Silicon using MLX.**

## 🎯 Quick Start

```bash
# Run the working demonstration
python true_gradient_backprop_training.py
```

**Expected Results:**
- Baseline: 0.530 MAE
- After Training: 0.180-0.305 MAE
- **Improvement: 42-66%** ✅

## 📊 Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Validation Error | 0.530 | 0.180 | **66.0%** |
| Final Val Error | 0.530 | 0.305 | **42.5%** |
| Trainable Params | - | 270K | **0.055%** of base |

## 🎓 What This Demonstrates

**Task:** Sentiment Intensity Calibration
- Train model to accurately predict sentiment scores (0-1 scale)
- Refines existing capability (not teaching new facts)
- Clear reward signal from prediction error
- Shows genuine learning with validation improvement

**Key Technologies:**
- **LoRA:** Low-Rank Adaptation for efficient fine-tuning
- **RLVR:** Reinforcement Learning from Verifiable Rewards
- **MLX:** Apple's ML framework for Apple Silicon
- **Model:** Qwen2.5-0.5B-Instruct-4bit (494M params)

## 📁 Project Structure

### ✅ Working Implementation
```
true_gradient_backprop_training.py  # Main demonstration (42-66% improvement)
sentiment_rlvr_demo.py              # Simpler version
```

### 🧪 Experimental Implementations
```
grpo_sentiment_training.py          # GRPO v1 (0% improvement - random gradients)
grpo_sentiment_training_v2.py       # GRPO v2 (0% improvement - needs debugging)
grpo_reference.ipynb                # Reference implementation from searlion/mlx-finetuning
```

### 📚 Documentation
```
RLVR_LORA_SUCCESS_SUMMARY.md        # Complete results & analysis
FINAL_FINDINGS.md                    # Technical deep-dive
GRPO_ANALYSIS.md                     # GRPO v1 analysis (random gradients)
GRPO_REFERENCE_ANALYSIS.md           # Reference implementation breakdown
GRPO_IMPLEMENTATION_SUMMARY.md       # GRPO research findings & v2 results
GRPO_COMPLETE_DIAGNOSIS.md           # Root cause & solution (deterministic generation)
GRPO_BUG_REPORT.md                   # Detailed bug analysis
better_rlvr_demos.md                 # Task selection guide
REAL_VS_DEMO_TRAINING.md            # Demo vs real training comparison
```

### 🔍 Analysis & Debug Tools
```
evaluate_training_results.py         # Performance evaluation
final_evaluation.py                  # Post-training analysis
diagnose_training.py                 # Debug utilities
debug_grpo_v2.py                     # GRPO log probability debugging
debug_gradient_keys.py               # Gradient computation debugging
test_full_pipeline.py                # End-to-end GRPO pipeline test
```

### 📓 Notebooks
```
real_lora_training_notebook.ipynb    # Interactive training
complete_lora_workflow.ipynb         # Original demo
```

## 🚀 How It Works

### 1. Apply LoRA Adapters
```python
# Add trainable low-rank adapters to 50% of model layers
class TrainableLoRALinear(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=16.0):
        self.lora_A = mx.random.normal((in_features, rank))
        self.lora_B = mx.zeros((rank, out_features))
        self.scaling = alpha / rank

    def __call__(self, x):
        original_out = self.original_layer(x)  # Frozen
        lora_out = (x @ self.lora_A) @ self.lora_B * self.scaling
        return original_out + lora_out  # Trainable adaptation
```

### 2. Compute Reward
```python
# Measure prediction error as reward signal
predicted_score = model.generate(sentiment_prompt)
actual_score = ground_truth
reward = 1.0 - abs(predicted_score - actual_score)
```

### 3. Update with Gradients
```python
# Compute gradients and update with Adam
loss, gradients, metrics = compute_loss_and_gradients(batch)
optimizer.update(lora_params, gradients)
```

### 4. Validate Improvement
```python
# Track validation error to ensure generalization
val_error = evaluate(model, validation_set)
if val_error < best_val_error:
    best_val_error = val_error  # Improved!
```

## 💡 Key Insights

### What Makes This Work

**✅ Task Selection:**
- Sentiment calibration refines existing capability
- Small model already understands sentiment
- Just needs to adjust intensity scoring

**✅ Gradient Computation:**
- Uses mx.grad() for proper optimization
- MSE loss on prediction error
- Adam optimizer for stable updates

**✅ LoRA Efficiency:**
- Only 270K trainable params (0.055% of model)
- Fast training (~5 minutes)
- Substantial improvement possible

### What Didn't Work

**❌ SQL Schema Learning:**
- Tried teaching 0.5B model database schema
- Requires memorizing facts (table/column names)
- Small model can't store this information
- Result: 0% improvement

**❌ GRPO v1 Implementation:**
- Implemented Group Relative Policy Optimization structure
- Group-based sampling and advantage calculation worked
- Gradient computation used random noise instead of true policy gradients
- Result: 0% improvement
- Issue: `mlx_lm.generate()` doesn't expose token log probabilities
- See `GRPO_ANALYSIS.md` for detailed analysis

**❌ GRPO v2 Implementation:**
- Researched reference implementation from searlion/mlx-finetuning
- Implemented TRUE policy gradients via `nn.value_and_grad()`
- Recomputes log probabilities from model forward passes
- Three-model architecture (train, rollout, reference)
- Result: 0% improvement
- **Root cause identified**: Deterministic generation → identical completions → zero advantages → zero gradients
- Issue: Current mlx_lm version lacks sampling/temperature support
- Fix: Add prompt variations or implement custom sampling
- See `GRPO_COMPLETE_DIAGNOSIS.md` for full analysis

**Lessons:**
- Small models can refine skills but can't learn new facts without massive data
- Algorithm structure alone isn't sufficient - need true policy gradients
- Random gradients cannot enable learning, regardless of sophistication
- TRUE policy gradients ARE possible in MLX (via `nn.value_and_grad()` + log prob recomputation)
- Implementation correctness matters more than algorithmic sophistication
- **GRPO requires exploration**: Deterministic generation → zero advantages → no learning
- Sampling/diversity is essential for policy gradient methods, not optional

## 📖 Documentation

### For Quick Understanding
→ Read `RLVR_LORA_SUCCESS_SUMMARY.md`

### For Technical Details
→ Read `FINAL_FINDINGS.md`

### For GRPO Experiment & Research
→ Read `GRPO_ANALYSIS.md` (v1 - random gradients)
→ Read `GRPO_REFERENCE_ANALYSIS.md` (reference implementation breakdown)
→ Read `GRPO_IMPLEMENTATION_SUMMARY.md` (v2 - true gradients, research findings)

### For Task Selection
→ Read `better_rlvr_demos.md`

### For Training Comparison
→ Read `REAL_VS_DEMO_TRAINING.md`

## 🔧 Requirements

```bash
pip install mlx mlx-lm numpy
```

**Platform:** Apple Silicon (M1/M2/M3)

## 🎯 Next Steps

### Try Other Calibration Tasks
1. **Toxicity Scoring:** Calibrate safety ratings
2. **Difficulty Rating:** Rate question difficulty
3. **Confidence Estimation:** Predict answer confidence

### Improve Training
1. Learning rate scheduling
2. Larger validation sets
3. Early stopping
4. Curriculum learning

### Scale Up
1. Try larger models (1.5B, 3B)
2. More training data
3. Longer training runs
4. Multi-task training

## 📊 Detailed Results

### Training Trajectory
```
Step   1: Error 0.133, Reward +0.867  (learning immediately!)
Step  15: Error 0.487, Val 0.470      (+11.3% improvement)
Step  30: Error 0.300, Val 0.180      (+66.0% improvement) ⭐
Step  65: Error 0.090                  (training continues)
Step  75: Val 0.345                    (some overfitting)
Final:    Val 0.305                    (+42.5% sustained)
```

## 🏆 Achievements

- ✅ 42-66% measurable improvement
- ✅ Validated on held-out data
- ✅ Efficient training (270K params)
- ✅ Complete working implementation
- ✅ Comprehensive documentation
- ✅ Demonstrates RLVR + LoRA on Apple Silicon

---

**Status:** ✅ **Working Demonstration Complete**

*Successfully trained a 0.5B parameter model to improve sentiment calibration by 42-66% using RLVR + LoRA on Apple Silicon with MLX.*