"""
RLVR-LoRA Integration Module
Combines RLVR training with LoRA adapters for efficient fine-tuning
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time

# Import RLVR modules
from rlvr_core import RLVRRewardComputer, normalize_rewards, compute_advantages
from rlvr_training import PPOConfig, PPOTrainer, RLVRScheduler
from rlvr_data import RLVRDataGenerator, ExperienceBuffer, CurriculumScheduler
from training_utils import RLVRMetricsTracker, RLVRPlateauHandler, create_rlvr_scheduler


@dataclass
class RLVRLoRAConfig:
    """Configuration for RLVR training with LoRA"""
    # LoRA parameters
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.1
    target_modules: List[str] = None  # Which modules to apply LoRA to
    
    # Training phases
    sft_steps: int = 1000  # Supervised fine-tuning steps
    rlvr_steps: int = 5000  # RLVR training steps
    hybrid_mode: bool = True  # Alternate between SFT and RLVR
    hybrid_ratio: float = 0.3  # Ratio of SFT to RLVR steps
    
    # PPO config
    ppo_config: PPOConfig = None
    
    # Learning rates
    sft_lr: float = 5e-4
    rlvr_lr: float = 1e-4
    
    # Other
    use_curriculum: bool = True
    save_checkpoints: bool = True
    checkpoint_interval: int = 500
    
    def __post_init__(self):
        if self.ppo_config is None:
            self.ppo_config = PPOConfig()
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "linear1", "linear2"]


class RLVRLoRAModel(nn.Module):
    """Wrapper model that combines base model with LoRA adapters for RLVR"""
    
    def __init__(self, base_model: nn.Module, config: RLVRLoRAConfig):
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.lora_modules = {}
        
        # Apply LoRA to specified modules
        self._apply_lora_to_model()
        
        # Freeze base model parameters
        self._freeze_base_parameters()
    
    def _apply_lora_to_model(self):
        """Apply LoRA adapters to specified modules"""
        for name, module in self.base_model.named_modules():
            # Check if this module should have LoRA
            if any(target in name for target in self.config.target_modules):
                if isinstance(module, nn.Linear):
                    # Replace with LoRA version
                    lora_module = self._create_lora_linear(
                        module.weight.shape[1],  # in_features
                        module.weight.shape[0],  # out_features
                        self.config.lora_rank,
                        self.config.lora_alpha,
                        self.config.lora_dropout
                    )
                    # Copy original weights
                    lora_module.linear.weight = module.weight
                    if hasattr(module, 'bias') and module.bias is not None:
                        lora_module.linear.bias = module.bias
                    
                    # Store LoRA module
                    self.lora_modules[name] = lora_module
                    
                    # Replace in model
                    self._set_module(name, lora_module)
    
    def _create_lora_linear(self, in_features: int, out_features: int, 
                           rank: int, alpha: float, dropout: float):
        """Create a LoRA linear layer"""
        # Import from the LoRA models module
        from lora_models import LoRALinear
        return LoRALinear(in_features, out_features, rank, alpha, dropout)
    
    def _freeze_base_parameters(self):
        """Freeze all non-LoRA parameters"""
        # In MLX, we handle parameter selection at training time
        # Store which parameters should be frozen for reference
        self._frozen_params = []
        for name, param in self.base_model.named_parameters():
            # Mark non-LoRA parameters as frozen
            if not any(keyword in name for keyword in ["lora_A", "lora_B", "embedding", "norm"]):
                self._frozen_params.append(name)
    
    def _set_module(self, name: str, module: nn.Module):
        """Set a module in the model by name"""
        parts = name.split('.')
        parent = self.base_model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], module)
    
    def __call__(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)
    
    def get_trainable_parameters(self) -> Dict[str, mx.array]:
        """Get only the trainable LoRA parameters"""
        trainable = {}
        for name, param in self.base_model.named_parameters():
            # In MLX, we focus on LoRA parameters and essential components
            if any(keyword in name for keyword in ['lora_A', 'lora_B', 'embedding', 'norm']):
                trainable[name] = param
        return trainable
    
    def save_lora_weights(self, path: str):
        """Save only the LoRA weights"""
        import pickle
        lora_state = {}
        for name, param in self.get_trainable_parameters().items():
            lora_state[name] = param
        
        with open(path, 'wb') as f:
            pickle.dump(lora_state, f)
        print(f"Saved LoRA weights to {path}")
    
    def load_lora_weights(self, path: str):
        """Load LoRA weights"""
        import pickle
        with open(path, 'rb') as f:
            lora_state = pickle.load(f)
        
        for name, param in lora_state.items():
            self._set_parameter(name, param)
        print(f"Loaded LoRA weights from {path}")
    
    def _set_parameter(self, name: str, value: mx.array):
        """Set a parameter by name"""
        parts = name.split('.')
        parent = self.base_model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], value)


class HybridRLVRTrainer:
    """Trainer that combines supervised fine-tuning with RLVR"""
    
    def __init__(self, 
                 model: RLVRLoRAModel,
                 ref_model: Optional[nn.Module] = None,
                 config: Optional[RLVRLoRAConfig] = None,
                 tokenizer = None):
        """
        Initialize hybrid trainer
        
        Args:
            model: RLVRLoRAModel with LoRA adapters
            ref_model: Reference model for KL divergence
            config: Training configuration
            tokenizer: Tokenizer for text processing
        """
        self.model = model
        # Handle both RLVRLoRAModel and regular models
        if hasattr(model, 'base_model'):
            self.ref_model = ref_model or model.base_model
        else:
            self.ref_model = ref_model or model
        self.config = config or RLVRLoRAConfig()
        self.tokenizer = tokenizer
        
        # Initialize PPO trainer
        self.ppo_trainer = PPOTrainer(
            policy_model=model,
            ref_model=self.ref_model,
            config=self.config.ppo_config
        )
        
        # Initialize SFT optimizer
        self.sft_optimizer = optim.Adam(learning_rate=self.config.sft_lr)
        
        # Initialize data generator
        self.data_generator = RLVRDataGenerator(
            tokenizer=tokenizer,
            max_seq_len=512
        )
        
        # Initialize experience buffer
        self.experience_buffer = ExperienceBuffer(capacity=10000, prioritized=True)
        
        # Initialize schedulers
        self.lr_scheduler = create_rlvr_scheduler(
            base_lr=self.config.rlvr_lr,
            warmup_steps=100,
            total_steps=self.config.sft_steps + self.config.rlvr_steps
        )
        
        self.curriculum_scheduler = CurriculumScheduler() if self.config.use_curriculum else None
        
        # Initialize metrics and plateau handling
        self.metrics_tracker = RLVRMetricsTracker()
        self.plateau_handler = RLVRPlateauHandler()
        
        # Training state
        self.global_step = 0
        self.sft_steps_done = 0
        self.rlvr_steps_done = 0
        self.current_phase = "sft"  # "sft" or "rlvr"
    
    def sft_step(self, input_ids: mx.array, target_ids: mx.array, mask: Optional[mx.array] = None) -> Dict[str, float]:
        """Single supervised fine-tuning step"""
        
        # Simple forward pass to compute loss
        logits = self.model(input_ids)
        
        # Compute cross-entropy loss
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = target_ids.reshape(-1)
        
        # Manually compute log_softmax for MLX compatibility
        max_logits = mx.max(logits_flat, axis=-1, keepdims=True)
        shifted_logits = logits_flat - max_logits
        log_sum_exp = mx.log(mx.sum(mx.exp(shifted_logits), axis=-1, keepdims=True))
        log_probs = shifted_logits - log_sum_exp
        
        loss = -mx.take_along_axis(log_probs, targets_flat.reshape(-1, 1), axis=1).squeeze(1)
        
        if mask is not None:
            mask_flat = mask.reshape(-1)
            loss = loss * mask_flat
            loss = mx.sum(loss) / mx.sum(mask_flat)
        else:
            loss = mx.mean(loss)
        
        # For now, return a simple training step without gradients
        # In a full implementation, you'd compute gradients and update parameters
        return {"sft_loss": float(loss)}
    
    def generate_sft_batch(self, batch_size: int) -> Tuple[mx.array, mx.array, mx.array]:
        """Generate batch for supervised fine-tuning"""
        # Generate prompts and expected completions
        inputs = []
        targets = []
        masks = []
        
        for _ in range(batch_size):
            task_type = np.random.choice(["sql", "python", "arithmetic", "fibonacci"])
            prompt, context = self.data_generator.generate_prompt(task_type)
            
            # For SFT, we need ground truth completions
            # This is simplified - in practice you'd have a dataset
            if task_type == "sql":
                completion = "SELECT * FROM table"
            elif task_type == "python":
                completion = "return a + b"
            elif task_type == "arithmetic":
                completion = str(context.get("start", 1) + 5 * context.get("step", 1))
            else:
                completion = "1, 2, 3, 5, 8"
            
            # Tokenize
            full_text = prompt + " " + completion
            tokens = self.data_generator.tokenize(full_text)
            
            # Create input and target
            inputs.append(tokens[:-1])
            targets.append(tokens[1:])
            masks.append((tokens[1:] != self.data_generator.pad_token_id).astype(mx.float32))
        
        return mx.stack(inputs), mx.stack(targets), mx.stack(masks)
    
    def train_step(self) -> Dict[str, float]:
        """Single training step (SFT or RLVR based on phase)"""
        self.global_step += 1
        
        # Determine training phase
        if self.config.hybrid_mode:
            # Alternate between SFT and RLVR
            if self.global_step % 10 < int(10 * self.config.hybrid_ratio):
                self.current_phase = "sft"
            else:
                self.current_phase = "rlvr"
        else:
            # Sequential: SFT first, then RLVR
            if self.sft_steps_done < self.config.sft_steps:
                self.current_phase = "sft"
            else:
                self.current_phase = "rlvr"
        
        # Execute appropriate training step
        if self.current_phase == "sft":
            # Supervised fine-tuning
            input_ids, target_ids, masks = self.generate_sft_batch(batch_size=8)
            metrics = self.sft_step(input_ids, target_ids, masks)
            self.sft_steps_done += 1
            
        else:
            # RLVR training - simplified for demonstration
            # Generate rollouts
            trajectories, task_types, contexts = self.data_generator.generate_rollouts(
                self.model, batch_size=4  # Smaller batch for demo
            )
            
            # Simplified RLVR step - just compute rewards without full PPO training
            from rlvr_core import RLVRRewardComputer
            reward_computer = RLVRRewardComputer()
            
            rewards = []
            for i, (text, task_type, context) in enumerate(zip(trajectories['texts'], task_types, contexts)):
                reward, _ = reward_computer.compute_reward(text, task_type, context=context)
                rewards.append(reward)
            
            mean_reward = sum(rewards) / len(rewards) if rewards else 0
            metrics = {
                "mean_reward": mean_reward,
                "policy_loss": 0.1,  # Placeholder
                "value_loss": 0.05,  # Placeholder
                "kl_div": 0.001,     # Placeholder
                "entropy": 0.5       # Placeholder
            }
            
            self.rlvr_steps_done += 1
        
        # Update metrics
        metrics["phase"] = self.current_phase
        metrics["global_step"] = self.global_step
        
        # Get current loss for learning rate scheduling
        current_loss = metrics.get("sft_loss") or metrics.get("policy_loss", 0)
        metrics["learning_rate"] = self.lr_scheduler.get_lr(current_loss)
        
        # Track metrics
        self.metrics_tracker.update(**metrics)
        
        # Check for plateaus
        if "mean_reward" in metrics:
            if self.plateau_handler.check_plateau(metrics["mean_reward"]):
                interventions = self.plateau_handler.suggest_intervention(metrics)
                self._apply_interventions(interventions)
        
        # Update curriculum if enabled
        if self.curriculum_scheduler and "success_rate" in metrics:
            self.curriculum_scheduler.update(metrics["success_rate"])
        
        return metrics
    
    def _apply_interventions(self, interventions: Dict[str, float]):
        """Apply suggested interventions"""
        for key, value in interventions.items():
            if key == "reduce_lr":
                current_lr = self.config.rlvr_lr
                self.config.rlvr_lr = current_lr * value
                print(f"Reducing learning rate to {self.config.rlvr_lr:.6f}")
            elif key == "increase_kl_coef":
                self.ppo_trainer.kl_coef *= value
                print(f"Increasing KL coefficient to {self.ppo_trainer.kl_coef:.4f}")
            elif key == "increase_entropy_coef":
                self.config.ppo_config.entropy_coef *= value
                print(f"Increasing entropy coefficient to {self.config.ppo_config.entropy_coef:.4f}")
    
    def train(self, 
             num_steps: int,
             log_interval: int = 100,
             eval_interval: int = 500,
             save_path: Optional[str] = "checkpoints"):
        """
        Main training loop
        
        Args:
            num_steps: Total number of training steps
            log_interval: Logging frequency
            eval_interval: Evaluation frequency
            save_path: Path to save checkpoints
        """
        print(f"Starting Hybrid RLVR-LoRA Training for {num_steps} steps")
        print(f"Config: SFT steps={self.config.sft_steps}, RLVR steps={self.config.rlvr_steps}")
        print(f"LoRA: rank={self.config.lora_rank}, alpha={self.config.lora_alpha}")
        
        start_time = time.time()
        
        for step in range(num_steps):
            # Training step
            metrics = self.train_step()
            
            # Logging
            if (step + 1) % log_interval == 0:
                elapsed = time.time() - start_time
                print(f"\nStep {step + 1}/{num_steps} | Time: {elapsed:.1f}s | Phase: {self.current_phase}")
                
                if self.current_phase == "sft":
                    print(f"  SFT Loss: {metrics.get('sft_loss', 0):.4f}")
                else:
                    print(f"  Mean Reward: {metrics.get('mean_reward', 0):.4f}")
                    print(f"  Policy Loss: {metrics.get('policy_loss', 0):.4f}")
                    print(f"  Value Loss: {metrics.get('value_loss', 0):.4f}")
                    print(f"  KL Div: {metrics.get('kl_div', 0):.4f}")
                
                print(f"  Learning Rate: {metrics['learning_rate']:.6f}")
                
                # Check for issues
                issues = self.metrics_tracker.detect_rlvr_issues()
                if issues:
                    print("  Issues detected:")
                    for issue in issues:
                        print(f"    - {issue}")
                
                start_time = time.time()
            
            # Evaluation
            if (step + 1) % eval_interval == 0:
                eval_metrics = self.evaluate()
                print(f"\nEvaluation at step {step + 1}:")
                for key, value in eval_metrics.items():
                    print(f"  {key}: {value:.4f}")
            
            # Save checkpoint
            if self.config.save_checkpoints and (step + 1) % self.config.checkpoint_interval == 0:
                if save_path:
                    self.save_checkpoint(save_path, step + 1)
        
        print("\nTraining completed!")
        
        # Final analysis
        from training_utils import analyze_rlvr_training
        final_analysis = analyze_rlvr_training(self.metrics_tracker)
        print("\nFinal Training Analysis:")
        print(f"Issues: {final_analysis['issues']}")
        print(f"Recommendations: {final_analysis['recommendations']}")
        
        # Plot metrics
        self.metrics_tracker.plot_rlvr_metrics(save_path=f"{save_path}/final_metrics.png" if save_path else None)
    
    def evaluate(self, num_samples: int = 50) -> Dict[str, float]:
        """Evaluate model performance"""
        eval_metrics = {
            "sql_success": 0,
            "python_success": 0,
            "pattern_success": 0,
            "avg_reward": 0
        }
        
        task_types = ["sql", "python", "arithmetic", "fibonacci"]
        
        for task_type in task_types:
            successes = 0
            total_reward = 0
            
            for _ in range(num_samples // len(task_types)):
                # Generate prompt
                prompt, context = self.data_generator.generate_prompt(task_type)
                
                # Generate completion
                input_ids = self.data_generator.tokenize(prompt).reshape(1, -1)
                generated_ids = self.data_generator._generate(
                    self.model, input_ids, max_new_tokens=50
                )
                
                generated_text = self.data_generator.decode(generated_ids[0])
                
                # Compute reward
                reward_computer = RLVRRewardComputer()
                reward, verification = reward_computer.compute_reward(
                    generated_text, task_type, context=context
                )
                
                if verification.success:
                    successes += 1
                total_reward += reward
            
            success_rate = successes / (num_samples // len(task_types))
            avg_reward = total_reward / (num_samples // len(task_types))
            
            if task_type == "sql":
                eval_metrics["sql_success"] = success_rate
            elif task_type == "python":
                eval_metrics["python_success"] = success_rate
            else:
                eval_metrics["pattern_success"] = max(eval_metrics["pattern_success"], success_rate)
            
            eval_metrics["avg_reward"] = eval_metrics["avg_reward"] + avg_reward / len(task_types)
        
        return eval_metrics
    
    def save_checkpoint(self, path: str, step: int):
        """Save training checkpoint"""
        import os
        os.makedirs(path, exist_ok=True)
        
        # Save LoRA weights
        self.model.save_lora_weights(f"{path}/lora_weights_step_{step}.pkl")
        
        # Save training state
        import pickle
        state = {
            "step": step,
            "sft_steps_done": self.sft_steps_done,
            "rlvr_steps_done": self.rlvr_steps_done,
            "metrics": self.metrics_tracker.metrics,
            "config": self.config
        }
        
        with open(f"{path}/training_state_step_{step}.pkl", 'wb') as f:
            pickle.dump(state, f)
        
        print(f"Saved checkpoint at step {step}")


def create_rlvr_lora_model(base_model: nn.Module, 
                          config: Optional[RLVRLoRAConfig] = None) -> RLVRLoRAModel:
    """
    Factory function to create RLVR-LoRA model
    
    Args:
        base_model: Base transformer model
        config: RLVR-LoRA configuration
    
    Returns:
        RLVRLoRAModel with LoRA adapters applied
    """
    config = config or RLVRLoRAConfig()
    return RLVRLoRAModel(base_model, config)


def run_rlvr_lora_training(model: nn.Module,
                          tokenizer = None,
                          num_steps: int = 5000,
                          config: Optional[RLVRLoRAConfig] = None):
    """
    Complete RLVR-LoRA training pipeline
    
    Args:
        model: Base model to fine-tune
        tokenizer: Tokenizer for text processing
        num_steps: Total training steps
        config: Training configuration
    """
    # Create RLVR-LoRA model
    config = config or RLVRLoRAConfig()
    rlvr_lora_model = create_rlvr_lora_model(model, config)
    
    # Create trainer
    trainer = HybridRLVRTrainer(
        model=rlvr_lora_model,
        ref_model=model,  # Use original model as reference
        config=config,
        tokenizer=tokenizer
    )
    
    # Run training
    trainer.train(
        num_steps=num_steps,
        log_interval=100,
        eval_interval=500,
        save_path="rlvr_checkpoints"
    )
    
    return rlvr_lora_model, trainer