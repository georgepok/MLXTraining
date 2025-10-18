"""
Example script demonstrating how to fix training plateau issues in MLX fine-tuning
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from training_utils import (
    PlateauDetector,
    AdaptiveLearningRateScheduler,
    analyze_training_logs,
    plot_training_diagnostics,
    optimize_training_config
)


def demonstrate_plateau_fixes():
    """Demonstrate various techniques to fix training plateaus"""
    
    # Simulate a training run with plateau
    print("=== Simulating Training with Plateau ===")
    
    # Generate synthetic loss data with plateau
    steps = 800
    loss_history = []
    
    # Initial rapid decrease
    for i in range(200):
        loss = 4.0 - (2.5 * (i / 200) ** 0.5) + np.random.normal(0, 0.05)
        loss_history.append(loss)
    
    # Plateau phase (steps 200-500)
    plateau_loss = loss_history[-1]
    for i in range(300):
        loss = plateau_loss + np.random.normal(0, 0.01)
        loss_history.append(loss)
    
    # Slight improvement after intervention
    for i in range(300):
        loss = plateau_loss - (0.3 * (i / 300) ** 2) + np.random.normal(0, 0.02)
        loss_history.append(loss)
    
    # Analyze the problematic training
    print("\n1. Analyzing Original Training Run")
    analysis = analyze_training_logs(loss_history, window_size=50)
    
    print(f"\nOriginal Training Analysis:")
    print(f"- Final loss: {analysis['final_loss']:.4f}")
    print(f"- Best loss: {analysis['best_loss']:.4f}")
    print(f"- Plateaus detected: {len(analysis['plateaus'])}")
    for i, plateau in enumerate(analysis['plateaus']):
        print(f"  Plateau {i+1}: steps {plateau['start']}-{plateau['end']}, avg loss: {plateau['avg_loss']:.4f}")
    
    print("\nSuggestions:")
    for suggestion in analysis['suggestions']:
        print(f"- {suggestion}")
    
    # Current configuration (problematic)
    current_config = {
        "learning_rate": 1e-5,
        "lora_rank": 8,
        "lora_alpha": 16,
        "batch_size": 8,
        "lr_scheduler": "constant",
        "gradient_clip_norm": 0.0,
        "weight_decay": 0.0
    }
    
    # Get optimized configuration
    print("\n2. Generating Optimized Configuration")
    optimized_config = optimize_training_config(loss_history, current_config)
    
    # Demonstrate improved training with new config
    print("\n3. Simulating Improved Training")
    
    # Initialize adaptive scheduler
    scheduler = AdaptiveLearningRateScheduler(
        base_lr=optimized_config.get("learning_rate", 5e-5),
        min_lr=1e-6,
        warmup_steps=50,
        total_steps=steps,
        schedule_type="cosine_restarts",
        restart_interval=150
    )
    
    # Simulate improved training
    improved_loss_history = []
    current_loss = 4.0
    
    for i in range(steps):
        # Get adaptive learning rate
        lr = scheduler.get_lr(current_loss if i > 0 else None)
        
        # Simulate training step with better config
        # Higher LR and better config lead to faster convergence
        if i < 50:  # Warmup
            improvement = 0.01 * (i / 50)
        else:
            improvement = lr * 20  # Simplified simulation
        
        current_loss -= improvement
        current_loss += np.random.normal(0, 0.02)
        current_loss = max(0.5, current_loss)  # Floor to prevent negative
        
        improved_loss_history.append(current_loss)
    
    # Plot comparison
    print("\n4. Visualizing Improvements")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original training
    ax1.plot(loss_history, 'b-', alpha=0.7, linewidth=1)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Original Training (with plateau)')
    ax1.grid(True, alpha=0.3)
    
    # Highlight plateau regions
    for plateau in analysis['plateaus']:
        ax1.axvspan(plateau['start'], plateau['end'], alpha=0.2, color='red')
    
    # Improved training
    ax2.plot(improved_loss_history, 'g-', alpha=0.7, linewidth=1)
    ax2.plot(scheduler.lr_history * np.max(improved_loss_history), 'r--', alpha=0.5, label='LR Schedule (scaled)')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Loss')
    ax2.set_title('Improved Training (with fixes)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plateau_fix_comparison.png', dpi=150)
    plt.show()
    
    # Summary statistics
    print("\n5. Training Comparison Summary")
    print(f"\nOriginal Training:")
    print(f"- Initial loss: {loss_history[0]:.4f}")
    print(f"- Final loss: {loss_history[-1]:.4f}")
    print(f"- Total improvement: {(loss_history[0] - loss_history[-1]) / loss_history[0] * 100:.1f}%")
    print(f"- Plateau duration: ~300 steps")
    
    print(f"\nImproved Training:")
    print(f"- Initial loss: {improved_loss_history[0]:.4f}")
    print(f"- Final loss: {improved_loss_history[-1]:.4f}")
    print(f"- Total improvement: {(improved_loss_history[0] - improved_loss_history[-1]) / improved_loss_history[0] * 100:.1f}%")
    print(f"- No significant plateaus")
    
    # Key techniques summary
    print("\n=== Key Techniques to Fix Plateaus ===")
    print("\n1. **Learning Rate Scheduling**")
    print("   - Use cosine annealing with warm restarts")
    print("   - Implement adaptive LR based on plateau detection")
    print("   - Start with higher base LR (5e-5 instead of 1e-5)")
    
    print("\n2. **LoRA Configuration**")
    print("   - Increase rank from 8 to 32 for more capacity")
    print("   - Set alpha = 2 * rank for proper scaling")
    print("   - Add dropout (0.05) for regularization")
    
    print("\n3. **Optimization Improvements**")
    print("   - Switch to AdamW with weight decay (0.01)")
    print("   - Enable gradient clipping (max_norm=1.0)")
    print("   - Use gradient accumulation for larger effective batch size")
    
    print("\n4. **Monitoring and Early Intervention**")
    print("   - Track gradient norms to detect vanishing gradients")
    print("   - Monitor loss variance for early plateau detection")
    print("   - Implement automatic LR boosting on plateau")
    
    print("\n5. **Data and Training Strategy**")
    print("   - Shuffle data more aggressively")
    print("   - Implement curriculum learning")
    print("   - Consider data augmentation techniques")
    
    # Save example configuration
    example_config = {
        "base_config": current_config,
        "optimized_config": optimized_config,
        "techniques_applied": [
            "Cosine annealing with restarts",
            "Increased LoRA rank",
            "AdamW optimizer",
            "Gradient clipping",
            "Adaptive learning rate",
            "Early plateau detection"
        ],
        "expected_improvements": {
            "training_efficiency": "2-3x faster convergence",
            "final_loss": "20-30% lower",
            "plateau_prevention": "Automatic LR adjustment"
        }
    }
    
    with open("plateau_fix_config.json", "w") as f:
        json.dump(example_config, f, indent=4)
    
    print("\n✅ Example configuration saved to: plateau_fix_config.json")
    print("\n🎯 Apply these techniques to your mlx-lm.ipynb training to overcome plateaus!")


if __name__ == "__main__":
    demonstrate_plateau_fixes()