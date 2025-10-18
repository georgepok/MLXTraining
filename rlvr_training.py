"""
RLVR Training Module with PPO (Proximal Policy Optimization) for MLX
Implements state-of-the-art RL training for language models
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time
from rlvr_core import RLVRRewardComputer, normalize_rewards, compute_advantages, compute_returns


@dataclass
class PPOConfig:
    """Configuration for PPO training"""
    # PPO hyperparameters
    clip_epsilon: float = 0.2
    value_clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Training parameters
    ppo_epochs: int = 4
    mini_batch_size: int = 8
    gamma: float = 0.99
    lam: float = 0.95
    
    # Learning rates
    policy_lr: float = 1e-4
    value_lr: float = 3e-4
    
    # KL penalty
    target_kl: float = 0.01
    kl_coef: float = 0.1
    adaptive_kl: bool = True
    
    # Other
    normalize_advantages: bool = True
    use_gae: bool = True
    gradient_checkpointing: bool = False


class ValueNetwork(nn.Module):
    """Value network for estimating state values"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x.squeeze(-1)


class PPOTrainer:
    """PPO trainer for RLVR fine-tuning"""
    
    def __init__(self, 
                 policy_model: nn.Module,
                 ref_model: Optional[nn.Module] = None,
                 config: Optional[PPOConfig] = None):
        """
        Initialize PPO trainer
        
        Args:
            policy_model: The model to train (with LoRA adapters)
            ref_model: Reference model for KL divergence (frozen)
            config: PPO configuration
        """
        self.policy_model = policy_model
        self.ref_model = ref_model or policy_model  # Use same model if no ref provided
        self.config = config or PPOConfig()
        
        # Freeze reference model
        if self.ref_model is not self.policy_model:
            for param in self.ref_model.parameters():
                param.stop_gradient = True
        
        # Initialize value network
        # Get hidden size from policy model
        hidden_size = self._get_hidden_size()
        self.value_network = ValueNetwork(hidden_size)
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(learning_rate=self.config.policy_lr)
        self.value_optimizer = optim.Adam(learning_rate=self.config.value_lr)
        
        # Initialize reward computer
        self.reward_computer = RLVRRewardComputer()
        
        # Training statistics
        self.training_stats = {
            "policy_losses": [],
            "value_losses": [],
            "entropy_losses": [],
            "kl_divergences": [],
            "rewards": [],
            "advantages": [],
            "clip_fractions": []
        }
        
        # Adaptive KL parameters
        self.kl_coef = self.config.kl_coef
    
    def _get_hidden_size(self) -> int:
        """Get hidden size from policy model"""
        # Try to get d_model attribute or default to 768
        if hasattr(self.policy_model, 'd_model'):
            return self.policy_model.d_model
        elif hasattr(self.policy_model, 'config') and hasattr(self.policy_model.config, 'hidden_size'):
            return self.policy_model.config.hidden_size
        else:
            return 768  # Default size
    
    def compute_logprobs_and_entropy(self, 
                                    logits: mx.array, 
                                    actions: mx.array,
                                    mask: Optional[mx.array] = None) -> Tuple[mx.array, mx.array]:
        """
        Compute log probabilities and entropy for actions
        
        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            actions: Taken actions [batch_size, seq_len]
            mask: Attention mask [batch_size, seq_len]
        
        Returns:
            Tuple of (log_probs, entropy)
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Compute log probabilities
        log_probs = mx.log_softmax(logits, axis=-1)
        
        # Get log probs for taken actions
        actions_flat = actions.reshape(-1)
        log_probs_flat = log_probs.reshape(-1, vocab_size)
        selected_log_probs = mx.take_along_axis(
            log_probs_flat, 
            actions_flat.reshape(-1, 1), 
            axis=1
        ).squeeze(-1)
        selected_log_probs = selected_log_probs.reshape(batch_size, seq_len)
        
        # Compute entropy
        probs = mx.exp(log_probs)
        entropy = -mx.sum(probs * log_probs, axis=-1)
        
        # Apply mask if provided
        if mask is not None:
            selected_log_probs = selected_log_probs * mask
            entropy = entropy * mask
            
            # Average over non-masked positions
            selected_log_probs = mx.sum(selected_log_probs, axis=1) / mx.sum(mask, axis=1)
            entropy = mx.sum(entropy, axis=1) / mx.sum(mask, axis=1)
        else:
            selected_log_probs = mx.mean(selected_log_probs, axis=1)
            entropy = mx.mean(entropy, axis=1)
        
        return selected_log_probs, entropy
    
    def compute_kl_divergence(self, 
                            policy_logits: mx.array, 
                            ref_logits: mx.array,
                            mask: Optional[mx.array] = None) -> mx.array:
        """
        Compute KL divergence between policy and reference model
        
        Args:
            policy_logits: Policy model logits
            ref_logits: Reference model logits
            mask: Attention mask
        
        Returns:
            KL divergence per sample
        """
        # Compute log probabilities
        policy_log_probs = mx.log_softmax(policy_logits, axis=-1)
        ref_log_probs = mx.log_softmax(ref_logits, axis=-1)
        
        # Compute KL divergence
        policy_probs = mx.exp(policy_log_probs)
        kl = mx.sum(policy_probs * (policy_log_probs - ref_log_probs), axis=-1)
        
        # Apply mask and average
        if mask is not None:
            kl = kl * mask
            kl = mx.sum(kl, axis=1) / mx.sum(mask, axis=1)
        else:
            kl = mx.mean(kl, axis=1)
        
        return kl
    
    def ppo_step(self, 
                states: mx.array,
                actions: mx.array,
                old_log_probs: mx.array,
                advantages: mx.array,
                returns: mx.array,
                mask: Optional[mx.array] = None) -> Dict[str, float]:
        """
        Single PPO optimization step
        
        Args:
            states: Input states [batch_size, seq_len]
            actions: Taken actions [batch_size, seq_len]
            old_log_probs: Log probabilities from rollout [batch_size]
            advantages: Computed advantages [batch_size]
            returns: Computed returns [batch_size]
            mask: Attention mask [batch_size, seq_len]
        
        Returns:
            Dictionary of loss components
        """
        
        def policy_loss_fn():
            # Forward pass through policy model
            logits = self.policy_model(states)
            
            # Compute log probs and entropy
            log_probs, entropy = self.compute_logprobs_and_entropy(logits, actions, mask)
            
            # Compute ratio
            ratio = mx.exp(log_probs - old_log_probs)
            
            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = mx.clip(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
            policy_loss = -mx.mean(mx.minimum(surr1, surr2))
            
            # Entropy loss
            entropy_loss = -self.config.entropy_coef * mx.mean(entropy)
            
            # KL penalty if using reference model
            kl_loss = 0
            if self.ref_model is not self.policy_model:
                with mx.no_grad():
                    ref_logits = self.ref_model(states)
                kl_div = self.compute_kl_divergence(logits, ref_logits, mask)
                kl_loss = self.kl_coef * mx.mean(kl_div)
            
            total_loss = policy_loss + entropy_loss + kl_loss
            
            # Store metrics for logging
            self.last_metrics = {
                "policy_loss": float(policy_loss),
                "entropy": float(mx.mean(entropy)),
                "kl_div": float(mx.mean(kl_div)) if kl_loss != 0 else 0,
                "clip_fraction": float(mx.mean((mx.abs(ratio - 1) > self.config.clip_epsilon).astype(mx.float32)))
            }
            
            return total_loss
        
        def value_loss_fn():
            # Get hidden states from policy model
            with mx.no_grad():
                hidden_states = self._get_hidden_states(states)
            
            # Predict values
            values = self.value_network(hidden_states)
            
            # Value loss (MSE)
            value_loss = self.config.value_loss_coef * mx.mean((values - returns) ** 2)
            
            return value_loss
        
        # Optimize policy
        policy_loss, policy_grads = mx.value_and_grad(policy_loss_fn)()
        self.policy_optimizer.update(self.policy_model, policy_grads)
        
        # Optimize value network
        value_loss, value_grads = mx.value_and_grad(value_loss_fn)()
        self.value_optimizer.update(self.value_network, value_grads)
        
        # Evaluate parameters
        mx.eval(self.policy_model.parameters())
        mx.eval(self.value_network.parameters())
        
        # Return metrics
        metrics = self.last_metrics.copy()
        metrics["value_loss"] = float(value_loss)
        metrics["total_loss"] = float(policy_loss) + float(value_loss)
        
        return metrics
    
    def _get_hidden_states(self, input_ids: mx.array) -> mx.array:
        """Get hidden states from model for value network"""
        # This is a simplified version - in practice, you'd extract
        # the actual hidden states from the model
        batch_size, seq_len = input_ids.shape
        hidden_size = self._get_hidden_size()
        
        # For now, use embedding layer output as hidden states
        if hasattr(self.policy_model, 'token_embedding'):
            hidden = self.policy_model.token_embedding(input_ids)
        else:
            # Fallback to random initialization
            hidden = mx.random.normal((batch_size, seq_len, hidden_size))
        
        # Average pool over sequence length
        return mx.mean(hidden, axis=1)
    
    def train_on_batch(self,
                      trajectories: Dict[str, Any],
                      task_types: List[str],
                      contexts: Optional[List[Dict]] = None) -> Dict[str, float]:
        """
        Train on a batch of trajectories
        
        Args:
            trajectories: Dictionary containing:
                - states: Input token ids [batch_size, seq_len]
                - actions: Generated token ids [batch_size, seq_len]
                - texts: Generated text strings
                - masks: Attention masks [batch_size, seq_len]
            task_types: Task type for each trajectory
            contexts: Optional context for reward computation
        
        Returns:
            Training metrics
        """
        states = trajectories['states']
        actions = trajectories['actions']
        texts = trajectories['texts']
        masks = trajectories.get('masks', None)
        
        batch_size = states.shape[0]
        
        # Get old log probabilities
        with mx.no_grad():
            old_logits = self.policy_model(states)
            old_log_probs, _ = self.compute_logprobs_and_entropy(old_logits, actions, masks)
            
            # Get values
            hidden_states = self._get_hidden_states(states)
            values = self.value_network(hidden_states)
            
            # Get KL divergence if using reference model
            kl_divs = None
            if self.ref_model is not self.policy_model:
                ref_logits = self.ref_model(states)
                kl_divs = self.compute_kl_divergence(old_logits, ref_logits, masks)
        
        # Compute rewards
        rewards_array, verifications = self.reward_computer.batch_compute_rewards(
            texts, task_types, contexts, kl_divs
        )
        
        # Compute advantages and returns
        if self.config.use_gae:
            advantages = compute_advantages(rewards_array, values, self.config.gamma, self.config.lam)
        else:
            returns = compute_returns(rewards_array, self.config.gamma)
            advantages = returns - values
        
        returns = advantages + values
        
        # Normalize advantages
        if self.config.normalize_advantages:
            advantages = normalize_rewards(advantages)
        
        # PPO training loop
        all_metrics = []
        
        for epoch in range(self.config.ppo_epochs):
            # Create mini-batches
            indices = mx.array(np.random.permutation(batch_size))
            
            for start_idx in range(0, batch_size, self.config.mini_batch_size):
                end_idx = min(start_idx + self.config.mini_batch_size, batch_size)
                batch_indices = indices[start_idx:end_idx]
                
                # Get mini-batch data
                mb_states = states[batch_indices]
                mb_actions = actions[batch_indices]
                mb_old_log_probs = old_log_probs[batch_indices]
                mb_advantages = advantages[batch_indices]
                mb_returns = returns[batch_indices]
                mb_masks = masks[batch_indices] if masks is not None else None
                
                # PPO step
                metrics = self.ppo_step(
                    mb_states, mb_actions, mb_old_log_probs,
                    mb_advantages, mb_returns, mb_masks
                )
                all_metrics.append(metrics)
            
            # Early stopping based on KL divergence
            if self.config.adaptive_kl and metrics["kl_div"] > self.config.target_kl * 1.5:
                print(f"Early stopping at epoch {epoch} due to high KL divergence: {metrics['kl_div']:.4f}")
                break
        
        # Adapt KL coefficient if needed
        if self.config.adaptive_kl:
            if metrics["kl_div"] > self.config.target_kl * 2:
                self.kl_coef *= 1.5
            elif metrics["kl_div"] < self.config.target_kl * 0.5:
                self.kl_coef *= 0.5
            self.kl_coef = np.clip(self.kl_coef, 0.001, 1.0)
        
        # Average metrics across all steps
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        # Add reward statistics
        avg_metrics["mean_reward"] = float(mx.mean(rewards_array))
        avg_metrics["max_reward"] = float(mx.max(rewards_array))
        avg_metrics["min_reward"] = float(mx.min(rewards_array))
        
        # Store in training stats
        for key, value in avg_metrics.items():
            if key in self.training_stats:
                self.training_stats[key].append(value)
        
        return avg_metrics
    
    def train(self,
             data_generator,
             num_iterations: int,
             rollout_size: int = 16,
             log_interval: int = 10,
             save_interval: int = 100,
             save_path: Optional[str] = None):
        """
        Main training loop
        
        Args:
            data_generator: Generator for creating prompts and contexts
            num_iterations: Number of training iterations
            rollout_size: Number of trajectories per iteration
            log_interval: Logging frequency
            save_interval: Model saving frequency
            save_path: Path to save model checkpoints
        """
        print(f"Starting PPO training for {num_iterations} iterations")
        print(f"Config: clip_eps={self.config.clip_epsilon}, "
              f"entropy_coef={self.config.entropy_coef}, "
              f"target_kl={self.config.target_kl}")
        
        start_time = time.time()
        
        for iteration in range(num_iterations):
            # Generate rollouts
            trajectories, task_types, contexts = data_generator.generate_rollouts(
                self.policy_model, rollout_size
            )
            
            # Train on batch
            metrics = self.train_on_batch(trajectories, task_types, contexts)
            
            # Logging
            if (iteration + 1) % log_interval == 0:
                elapsed = time.time() - start_time
                print(f"\nIteration {iteration + 1}/{num_iterations} | Time: {elapsed:.1f}s")
                print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
                print(f"  Value Loss: {metrics['value_loss']:.4f}")
                print(f"  Mean Reward: {metrics['mean_reward']:.4f}")
                print(f"  KL Div: {metrics['kl_div']:.4f}")
                print(f"  Clip Fraction: {metrics['clip_fraction']:.3f}")
                print(f"  Entropy: {metrics['entropy']:.4f}")
                
                if self.config.adaptive_kl:
                    print(f"  KL Coef: {self.kl_coef:.4f}")
                
                start_time = time.time()
            
            # Save checkpoint
            if save_path and (iteration + 1) % save_interval == 0:
                self.save_checkpoint(save_path, iteration)
        
        print("\nTraining completed!")
        return self.training_stats
    
    def save_checkpoint(self, path: str, iteration: int):
        """Save model checkpoint"""
        checkpoint = {
            "iteration": iteration,
            "policy_model": self.policy_model.state_dict() if hasattr(self.policy_model, 'state_dict') else None,
            "value_network": self.value_network.state_dict() if hasattr(self.value_network, 'state_dict') else None,
            "kl_coef": self.kl_coef,
            "training_stats": self.training_stats
        }
        
        import pickle
        checkpoint_path = f"{path}/checkpoint_{iteration}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"Saved checkpoint to {checkpoint_path}")


class RLVRScheduler:
    """Learning rate scheduler for RLVR training"""
    
    def __init__(self, 
                 base_lr: float,
                 warmup_steps: int = 100,
                 decay_steps: int = 1000,
                 min_lr: float = 1e-6):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.min_lr = min_lr
        self.current_step = 0
    
    def get_lr(self, metrics: Optional[Dict[str, float]] = None) -> float:
        """Get current learning rate based on step and optional metrics"""
        step = self.current_step
        
        # Warmup phase
        if step < self.warmup_steps:
            lr = self.base_lr * (step / self.warmup_steps)
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / self.decay_steps
            progress = min(1.0, progress)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        # Adjust based on metrics if provided
        if metrics:
            # Reduce LR if KL divergence is too high
            if metrics.get('kl_div', 0) > 0.02:
                lr *= 0.5
            # Increase LR if learning is too slow
            elif metrics.get('policy_loss', float('inf')) > metrics.get('initial_loss', float('inf')) * 0.9:
                lr *= 1.1
        
        self.current_step += 1
        return np.clip(lr, self.min_lr, self.base_lr)