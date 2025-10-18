"""
Training utilities for MLX fine-tuning with plateau prevention and optimization strategies
"""

import math
import numpy as np
import mlx.core as mx
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import json
import matplotlib.pyplot as plt


class PlateauDetector:
    """Detects training plateaus and suggests interventions"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.plateau_count = 0
        self.loss_history = []
        
    def update(self, loss: float) -> bool:
        """Update with new loss value and check for plateau"""
        self.loss_history.append(loss)
        
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.plateau_count = 0
            return False
        else:
            self.plateau_count += 1
            return self.plateau_count >= self.patience
    
    def get_suggestions(self) -> List[str]:
        """Get suggestions based on plateau analysis"""
        suggestions = []
        
        if len(self.loss_history) < 10:
            return ["Need more training data to analyze"]
        
        # Analyze recent loss trend
        recent_losses = self.loss_history[-10:]
        loss_std = np.std(recent_losses)
        loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
        
        if loss_std < 0.001:
            suggestions.append("Loss variance very low - consider increasing learning rate")
        
        if abs(loss_trend) < 0.0001:
            suggestions.append("Loss completely flat - try learning rate restart")
            
        if self.plateau_count > self.patience * 2:
            suggestions.append("Extended plateau - consider changing LoRA rank or architecture")
            
        return suggestions


class AdaptiveLearningRateScheduler:
    """Advanced learning rate scheduler with plateau-based adjustments"""
    
    def __init__(self, 
                 base_lr: float,
                 min_lr: float,
                 warmup_steps: int,
                 total_steps: int,
                 schedule_type: str = "cosine_restarts",
                 restart_interval: int = 100):
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.schedule_type = schedule_type
        self.restart_interval = restart_interval
        self.current_step = 0
        self.plateau_detector = PlateauDetector()
        self.lr_history = []
        
    def get_lr(self, loss: Optional[float] = None) -> float:
        """Get learning rate with optional plateau-based adjustment"""
        step = self.current_step
        
        # Warmup phase
        if step < self.warmup_steps:
            lr = self.base_lr * (step / self.warmup_steps)
        else:
            if self.schedule_type == "cosine":
                lr = self._cosine_schedule(step)
            elif self.schedule_type == "cosine_restarts":
                lr = self._cosine_with_restarts(step)
            elif self.schedule_type == "exponential":
                lr = self._exponential_schedule(step)
            else:
                lr = self.base_lr
        
        # Check for plateau and adjust if needed
        if loss is not None and self.plateau_detector.update(loss):
            # Plateau detected - boost learning rate temporarily
            lr = min(lr * 2, self.base_lr)
            print(f"Plateau detected at step {step}, boosting LR to {lr:.2e}")
        
        self.current_step += 1
        self.lr_history.append(lr)
        return lr
    
    def _cosine_schedule(self, step: int) -> float:
        """Standard cosine annealing"""
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(1.0, progress)
        return self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
    
    def _cosine_with_restarts(self, step: int) -> float:
        """Cosine annealing with warm restarts"""
        cycle_step = (step - self.warmup_steps) % self.restart_interval
        cycle_progress = cycle_step / self.restart_interval
        
        # Decay the base LR over cycles
        num_cycles = (step - self.warmup_steps) // self.restart_interval
        cycle_base_lr = self.base_lr * (0.8 ** num_cycles)
        
        return self.min_lr + (cycle_base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * cycle_progress))
    
    def _exponential_schedule(self, step: int) -> float:
        """Exponential decay"""
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return self.base_lr * (self.min_lr / self.base_lr) ** progress


def analyze_training_logs(loss_history: List[float], window_size: int = 50) -> Dict[str, any]:
    """Analyze training logs to identify issues and suggest improvements"""
    
    analysis = {
        "total_steps": len(loss_history),
        "final_loss": loss_history[-1] if loss_history else None,
        "best_loss": min(loss_history) if loss_history else None,
        "plateaus": [],
        "suggestions": []
    }
    
    if len(loss_history) < window_size:
        analysis["suggestions"].append("Not enough training data for comprehensive analysis")
        return analysis
    
    # Detect plateaus
    for i in range(window_size, len(loss_history), window_size // 2):
        window = loss_history[i-window_size:i]
        window_std = np.std(window)
        window_trend = np.polyfit(range(len(window)), window, 1)[0]
        
        if window_std < 0.01 and abs(window_trend) < 0.0001:
            analysis["plateaus"].append({
                "start": i - window_size,
                "end": i,
                "avg_loss": np.mean(window),
                "std": window_std
            })
    
    # Generate suggestions based on analysis
    if analysis["plateaus"]:
        total_plateau_steps = sum(p["end"] - p["start"] for p in analysis["plateaus"])
        plateau_ratio = total_plateau_steps / len(loss_history)
        
        if plateau_ratio > 0.3:
            analysis["suggestions"].append(f"Training plateaued for {plateau_ratio*100:.1f}% of training")
            analysis["suggestions"].append("Consider: Higher LoRA rank, learning rate scheduling, or data augmentation")
    
    # Check for oscillations
    recent_losses = loss_history[-window_size:]
    loss_diffs = np.diff(recent_losses)
    sign_changes = np.sum(np.diff(np.sign(loss_diffs)) != 0)
    
    if sign_changes > window_size * 0.8:
        analysis["suggestions"].append("High oscillation detected - reduce learning rate")
    
    # Check improvement rate
    early_avg = np.mean(loss_history[:window_size])
    late_avg = np.mean(loss_history[-window_size:])
    improvement = (early_avg - late_avg) / early_avg
    
    if improvement < 0.1:
        analysis["suggestions"].append(f"Low overall improvement ({improvement*100:.1f}%) - check data quality and model capacity")
    
    return analysis


def plot_training_diagnostics(metrics: Dict[str, List[Tuple[int, float]]], save_path: Optional[str] = None):
    """Create comprehensive training diagnostic plots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # 1. Loss curves
    if "train_losses" in metrics:
        steps, losses = zip(*metrics["train_losses"])
        axes[0].plot(steps, losses, 'b-', alpha=0.7, label='Train Loss')
        
        # Add smoothed curve
        if len(losses) > 20:
            window = min(20, len(losses) // 10)
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            axes[0].plot(steps[window-1:], smoothed, 'r-', linewidth=2, label='Smoothed')
        
        axes[0].set_xlabel('Steps')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss Evolution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # 2. Loss improvement rate
    if "train_losses" in metrics and len(metrics["train_losses"]) > 20:
        steps, losses = zip(*metrics["train_losses"])
        improvement_rates = []
        window = 10
        
        for i in range(window, len(losses)):
            rate = (losses[i-window] - losses[i]) / window
            improvement_rates.append(rate)
        
        axes[1].plot(steps[window:], improvement_rates, 'g-')
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Steps')
        axes[1].set_ylabel('Improvement Rate')
        axes[1].set_title('Loss Improvement Rate (10-step window)')
        axes[1].grid(True, alpha=0.3)
    
    # 3. Learning rate schedule
    if "learning_rates" in metrics:
        steps, lrs = zip(*metrics["learning_rates"])
        axes[2].plot(steps, lrs, 'purple')
        axes[2].set_xlabel('Steps')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_yscale('log')
    
    # 4. Gradient norms
    if "gradient_norms" in metrics:
        steps, norms = zip(*metrics["gradient_norms"])
        axes[3].plot(steps, norms, 'red')
        axes[3].set_xlabel('Steps')
        axes[3].set_ylabel('Gradient Norm')
        axes[3].set_title('Gradient Norm Evolution')
        axes[3].grid(True, alpha=0.3)
        axes[3].set_yscale('log')
    
    # 5. Validation vs Training
    if "train_losses" in metrics and "val_losses" in metrics:
        train_steps, train_losses = zip(*metrics["train_losses"])
        val_steps, val_losses = zip(*metrics["val_losses"])
        
        axes[4].plot(train_steps, train_losses, 'b-', alpha=0.7, label='Train')
        axes[4].plot(val_steps, val_losses, 'orange', marker='o', label='Validation')
        axes[4].set_xlabel('Steps')
        axes[4].set_ylabel('Loss')
        axes[4].set_title('Train vs Validation Loss')
        axes[4].legend()
        axes[4].grid(True, alpha=0.3)
    
    # 6. Loss distribution histogram
    if "train_losses" in metrics:
        _, losses = zip(*metrics["train_losses"])
        axes[5].hist(losses, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[5].set_xlabel('Loss Value')
        axes[5].set_ylabel('Frequency')
        axes[5].set_title('Loss Distribution')
        axes[5].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Diagnostic plots saved to: {save_path}")
    
    plt.show()


def suggest_hyperparameters(current_config: Dict, training_analysis: Dict) -> Dict:
    """Suggest improved hyperparameters based on training analysis"""
    
    suggestions = dict(current_config)
    
    # Adjust LoRA rank based on plateau analysis
    if training_analysis.get("plateaus"):
        current_rank = current_config.get("lora_rank", 8)
        suggestions["lora_rank"] = min(current_rank * 2, 64)
        suggestions["lora_alpha"] = suggestions["lora_rank"] * 2
    
    # Adjust learning rate based on oscillation
    if "High oscillation" in str(training_analysis.get("suggestions", [])):
        current_lr = current_config.get("learning_rate", 1e-4)
        suggestions["learning_rate"] = current_lr * 0.5
    
    # Add scheduling if plateaus detected
    if training_analysis.get("plateaus"):
        suggestions["lr_scheduler"] = "cosine_restarts"
        suggestions["warmup_ratio"] = 0.1
    
    # Increase batch size if loss is stable but slow
    if "Low overall improvement" in str(training_analysis.get("suggestions", [])):
        current_batch = current_config.get("batch_size", 8)
        suggestions["batch_size"] = min(current_batch * 2, 32)
        suggestions["gradient_accumulation_steps"] = 4
    
    return suggestions


# Example usage function
def optimize_training_config(loss_history: List[float], current_config: Dict) -> Dict:
    """Analyze training and return optimized configuration"""
    
    # Analyze current training
    analysis = analyze_training_logs(loss_history)
    
    # Get suggested hyperparameters
    new_config = suggest_hyperparameters(current_config, analysis)
    
    # Print analysis results
    print("Training Analysis:")
    print(f"- Total steps: {analysis['total_steps']}")
    print(f"- Plateaus detected: {len(analysis['plateaus'])}")
    for suggestion in analysis['suggestions']:
        print(f"- {suggestion}")
    
    print("\nSuggested Configuration Changes:")
    for key, value in new_config.items():
        if key in current_config and current_config[key] != value:
            print(f"- {key}: {current_config[key]} → {value}")
    
    return new_config


# RLVR-specific utilities
class RLVRMetricsTracker:
    """Track metrics specific to RLVR training"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = {
            "rewards": [],
            "success_rates": [],
            "kl_divergences": [],
            "entropy": [],
            "clip_fractions": [],
            "value_losses": [],
            "policy_losses": [],
            "verification_scores": []
        }
        self.step_count = 0
    
    def update(self, **kwargs):
        """Update metrics with new values"""
        self.step_count += 1
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)
                # Keep only window_size most recent values
                if len(self.metrics[key]) > self.window_size * 10:
                    self.metrics[key] = self.metrics[key][-self.window_size * 10:]
    
    def get_recent_stats(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for recent values of a metric"""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return {}
        
        recent = self.metrics[metric_name][-self.window_size:]
        return {
            "mean": np.mean(recent),
            "std": np.std(recent),
            "min": np.min(recent),
            "max": np.max(recent),
            "median": np.median(recent)
        }
    
    def get_trend(self, metric_name: str, window: int = 50) -> float:
        """Get trend (slope) of metric over recent window"""
        if metric_name not in self.metrics or len(self.metrics[metric_name]) < window:
            return 0.0
        
        recent = self.metrics[metric_name][-window:]
        x = np.arange(len(recent))
        trend = np.polyfit(x, recent, 1)[0]
        return trend
    
    def detect_rlvr_issues(self) -> List[str]:
        """Detect common RLVR training issues"""
        issues = []
        
        # Check reward collapse
        reward_stats = self.get_recent_stats("rewards")
        if reward_stats and reward_stats["std"] < 0.01:
            issues.append("Reward collapse detected - rewards have very low variance")
        
        # Check KL divergence
        kl_stats = self.get_recent_stats("kl_divergences")
        if kl_stats and kl_stats["mean"] > 0.05:
            issues.append("High KL divergence - policy diverging from reference")
        
        # Check entropy collapse
        entropy_stats = self.get_recent_stats("entropy")
        if entropy_stats and entropy_stats["mean"] < 0.1:
            issues.append("Low entropy - policy becoming too deterministic")
        
        # Check clip fraction
        clip_stats = self.get_recent_stats("clip_fractions")
        if clip_stats and clip_stats["mean"] > 0.3:
            issues.append("High clip fraction - consider reducing learning rate")
        
        # Check success rate trend
        success_trend = self.get_trend("success_rates")
        if success_trend < -0.001:
            issues.append("Declining success rate - performance degrading")
        
        return issues
    
    def plot_rlvr_metrics(self, save_path: Optional[str] = None):
        """Create RLVR-specific diagnostic plots"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        # Plot each metric
        metric_configs = [
            ("rewards", "Rewards", "green"),
            ("success_rates", "Success Rate", "blue"),
            ("kl_divergences", "KL Divergence", "red"),
            ("entropy", "Entropy", "purple"),
            ("clip_fractions", "Clip Fraction", "orange"),
            ("value_losses", "Value Loss", "brown"),
            ("policy_losses", "Policy Loss", "pink"),
            ("verification_scores", "Verification Score", "cyan")
        ]
        
        for idx, (metric_name, title, color) in enumerate(metric_configs):
            if idx < len(axes) and metric_name in self.metrics and self.metrics[metric_name]:
                data = self.metrics[metric_name]
                axes[idx].plot(data, color=color, alpha=0.7)
                
                # Add rolling average
                if len(data) > 20:
                    window = min(50, len(data) // 5)
                    rolling_avg = np.convolve(data, np.ones(window)/window, mode='valid')
                    axes[idx].plot(range(window-1, len(data)), rolling_avg, 
                                 color=color, linewidth=2, label='Rolling Avg')
                
                axes[idx].set_title(title)
                axes[idx].set_xlabel('Steps')
                axes[idx].set_ylabel(title)
                axes[idx].grid(True, alpha=0.3)
                axes[idx].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"RLVR metrics plot saved to: {save_path}")
        
        plt.show()


class RLVRPlateauHandler:
    """Handle plateaus specific to RLVR training"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_reward = -float('inf')
        self.plateau_count = 0
        self.interventions_applied = []
    
    def check_plateau(self, current_reward: float) -> bool:
        """Check if training has plateaued based on reward"""
        if current_reward > self.best_reward + self.min_delta:
            self.best_reward = current_reward
            self.plateau_count = 0
            return False
        else:
            self.plateau_count += 1
            return self.plateau_count >= self.patience
    
    def suggest_intervention(self, metrics: Dict[str, float]) -> Dict[str, any]:
        """Suggest intervention based on current metrics"""
        interventions = {}
        
        # Check various metrics and suggest interventions
        if metrics.get('kl_divergence', 0) > 0.03:
            interventions['reduce_lr'] = 0.5
            interventions['increase_kl_coef'] = 1.5
        
        if metrics.get('entropy', float('inf')) < 0.5:
            interventions['increase_entropy_coef'] = 2.0
            interventions['increase_temperature'] = 1.2
        
        if metrics.get('clip_fraction', 0) > 0.2:
            interventions['reduce_clip_epsilon'] = 0.8
            interventions['reduce_lr'] = 0.7
        
        if metrics.get('success_rate', 0) < 0.3:
            interventions['reduce_difficulty'] = 0.8
            interventions['increase_exploration'] = 1.5
        
        # Track applied interventions
        self.interventions_applied.append({
            "step": metrics.get('step', 0),
            "interventions": interventions,
            "metrics": metrics.copy()
        })
        
        return interventions
    
    def reset(self):
        """Reset plateau detection"""
        self.plateau_count = 0
        self.best_reward = -float('inf')


def create_rlvr_scheduler(base_lr: float, 
                         warmup_steps: int = 100,
                         total_steps: int = 10000,
                         schedule_type: str = "cosine_with_restarts") -> AdaptiveLearningRateScheduler:
    """Create a learning rate scheduler optimized for RLVR training"""
    
    scheduler = AdaptiveLearningRateScheduler(
        base_lr=base_lr,
        min_lr=base_lr * 0.01,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        schedule_type=schedule_type,
        restart_interval=total_steps // 10  # Restart every 10% of training
    )
    
    # Modify plateau detector for RLVR
    scheduler.plateau_detector = PlateauDetector(patience=20, min_delta=0.0001)
    
    return scheduler


def analyze_rlvr_training(metrics_tracker: RLVRMetricsTracker) -> Dict[str, any]:
    """Comprehensive analysis of RLVR training progress"""
    
    analysis = {
        "issues": metrics_tracker.detect_rlvr_issues(),
        "trends": {},
        "recommendations": []
    }
    
    # Analyze trends
    for metric in ["rewards", "success_rates", "kl_divergences", "entropy"]:
        trend = metrics_tracker.get_trend(metric)
        analysis["trends"][metric] = {
            "direction": "increasing" if trend > 0 else "decreasing",
            "rate": abs(trend)
        }
    
    # Generate recommendations
    if "Reward collapse" in str(analysis["issues"]):
        analysis["recommendations"].append("Increase exploration or entropy coefficient")
    
    if "High KL divergence" in str(analysis["issues"]):
        analysis["recommendations"].append("Reduce learning rate or increase KL penalty")
    
    if analysis["trends"]["rewards"]["direction"] == "decreasing":
        analysis["recommendations"].append("Consider reverting to earlier checkpoint")
    
    if analysis["trends"]["success_rates"]["rate"] < 0.001:
        analysis["recommendations"].append("Training progress is slow - adjust hyperparameters")
    
    return analysis