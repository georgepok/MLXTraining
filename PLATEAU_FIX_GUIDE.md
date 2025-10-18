# Quick Guide: Fixing Training Plateau in MLX Fine-tuning

## Immediate Actions When Loss Flattens (~300 iterations)

### 1. **Quick Config Changes** (Try First)
```python
# Increase LoRA rank and alpha
lora_config = {
    "num_layers": 16,  # was 8
    "lora_parameters": {
        "rank": 32,  # was 8
        "scale": 64.0,  # was 20.0 (use 2x rank)
        "dropout": 0.05,  # was 0.0
    },
}

# Boost learning rate with scheduling
optimizer = optim.AdamW(
    learning_rate=5e-5,  # was 1e-5
    weight_decay=0.01    # was 0.0
)
```

### 2. **Learning Rate Schedule** (Most Effective)
```python
# Add cosine annealing with warm restarts
from training_utils import AdaptiveLearningRateScheduler

scheduler = AdaptiveLearningRateScheduler(
    base_lr=5e-5,
    min_lr=1e-6,
    warmup_steps=50,
    total_steps=800,
    schedule_type="cosine_restarts",
    restart_interval=150
)

# Use in training loop
current_lr = scheduler.get_lr(loss)
optimizer.learning_rate = current_lr
```

### 3. **Gradient Clipping** (Stability)
```python
# Prevent gradient explosion
gradient_clip_norm = 1.0

# In training loop:
grad_norm = compute_gradient_norm(grads)
if grad_norm > gradient_clip_norm:
    scale = gradient_clip_norm / grad_norm
    grads = tree_map(lambda g: g * scale, grads)
```

### 4. **Training Arguments Update**
```python
training_args = TrainingArgs(
    adapter_file=adapter_file_path,
    iters=800,
    steps_per_eval=25,  # was 50 - more frequent evaluation
    # Add these if using custom training:
    gradient_accumulation_steps=4,  # Effective larger batch
    mixed_precision=True,  # Faster training
)
```

## Diagnostic Commands

### Check if You're in a Plateau:
```python
# Look at recent loss variance
recent_losses = trainer.train_losses[-50:]
loss_std = np.std(recent_losses)
if loss_std < 0.01:
    print("WARNING: Training has plateaued!")
```

### Visualize Training:
```python
from training_utils import plot_training_diagnostics

metrics = {
    "train_losses": [(i, loss) for i, loss in enumerate(trainer.train_losses)],
    "val_losses": [(i, loss) for i, loss in enumerate(trainer.val_losses)]
}
plot_training_diagnostics(metrics, save_path="training_analysis.png")
```

## Common Plateau Patterns & Solutions

| Symptom | Cause | Solution |
|---------|-------|----------|
| Loss stops decreasing after 200-300 steps | LR too low | Increase base LR to 5e-5 or 1e-4 |
| Loss oscillates around same value | LR too high | Reduce LR or add scheduling |
| Very small loss changes (<0.001) | LoRA rank too low | Increase rank to 16 or 32 |
| Validation loss increases | Overfitting | Add dropout, reduce LR |
| Gradient norm near zero | Vanishing gradients | Check model architecture, increase LR |

## Emergency Fixes (When Nothing Else Works)

### 1. **Learning Rate Restart**
```python
# Boost LR temporarily
if plateau_detected:
    optimizer.learning_rate = original_lr * 2
    # Train for 50 steps
    # Then resume normal schedule
```

### 2. **Progressive LoRA Unfreezing**
```python
# Start with fewer layers, gradually add more
if step > 400 and loss_plateaued:
    linear_to_lora_layers(model, num_layers=24, lora_params)
```

### 3. **Data Augmentation**
```python
# Add noise to inputs to increase diversity
def augment_batch(batch):
    noise = mx.random.normal(batch.shape) * 0.01
    return batch + noise
```

## Recommended Training Pipeline

```python
# 1. Use enhanced notebook
# Run: mlx_lm_enhanced.ipynb

# 2. Monitor with utilities
from training_utils import PlateauDetector, analyze_training_logs

detector = PlateauDetector(patience=5)
for step in training_loop:
    if detector.update(loss):
        print("Plateau detected! Applying fixes...")
        # Apply emergency fixes

# 3. Analyze results
analysis = analyze_training_logs(loss_history)
print(analysis['suggestions'])
```

## Prevention is Better Than Cure

Start training with these settings to avoid plateaus:
- LoRA rank: 16-32 (not 8)
- Learning rate: 5e-5 with cosine scheduling
- Weight decay: 0.01
- Gradient clipping: 1.0
- Evaluation frequency: Every 25 steps
- Early stopping patience: 10 evaluations

## Files Created for You:
1. `mlx_lm_enhanced.ipynb` - Full enhanced training notebook
2. `training_utils.py` - Utility functions for plateau detection
3. `fix_plateau_example.py` - Demonstration of fixes
4. `PLATEAU_FIX_GUIDE.md` - This quick reference

Run the enhanced notebook to avoid plateau issues from the start!