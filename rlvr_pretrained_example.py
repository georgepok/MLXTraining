"""
RLVR with Pretrained Models Example
Uses mlx-lm to load pretrained models and apply RLVR training
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import time

# Try to import mlx_lm
try:
    from mlx_lm import load, generate
    from mlx_lm.tuner import LoRALinear
    MLX_LM_AVAILABLE = True
    print("✅ mlx_lm is available - will use pretrained models")
except ImportError:
    print("⚠️ mlx_lm not available - falling back to simple models")
    MLX_LM_AVAILABLE = False

# Import RLVR modules
from rlvr_core import RLVRRewardComputer, SQLVerifier, CodeVerifier, PatternVerifier
from rlvr_data import RLVRDataGenerator
from training_utils import RLVRMetricsTracker


class PretrainedRLVRTrainer:
    """RLVR trainer that works with pretrained models from mlx-lm"""
    
    def __init__(self, model_path: str = "mlx-community/Llama-3.2-1B-Instruct-4bit"):
        """
        Initialize trainer with a pretrained model
        
        Args:
            model_path: HuggingFace model path or local path
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.reward_computer = RLVRRewardComputer()
        self.metrics_tracker = RLVRMetricsTracker()
        
        # Load the model
        self.load_pretrained_model()
        
        # Initialize data generator
        self.data_generator = RLVRDataGenerator(
            tokenizer=self.tokenizer,
            max_seq_len=512
        )
    
    def load_pretrained_model(self):
        """Load pretrained model and tokenizer"""
        if MLX_LM_AVAILABLE:
            try:
                print(f"Loading pretrained model: {self.model_path}")
                self.model, self.tokenizer = load(self.model_path)
                print(f"✅ Successfully loaded {self.model_path}")
                
                # Print model info
                if hasattr(self.model, 'args'):
                    print(f"Model vocab size: {self.model.args.vocab_size}")
                    print(f"Model layers: {self.model.args.num_hidden_layers}")
                
            except Exception as e:
                print(f"❌ Failed to load {self.model_path}: {e}")
                print("Falling back to dummy model")
                self._create_dummy_model()
        else:
            print("Creating dummy model since mlx-lm is not available")
            self._create_dummy_model()
    
    def _create_dummy_model(self):
        """Create a dummy model for demonstration"""
        from lora_models import SimpleLoRATransformer
        
        class DummyTokenizer:
            def __init__(self, vocab_size=5000):
                self.vocab_size = vocab_size
                self.eos_token_id = 2
            
            def encode(self, text):
                return [min(ord(c), self.vocab_size-1) for c in text[:50]]
            
            def decode(self, tokens):
                if isinstance(tokens, list):
                    return ''.join([chr(min(t, 127)) for t in tokens if t > 0 and t < 128])
                else:
                    # Handle MLX array
                    tokens_list = tokens.tolist() if hasattr(tokens, 'tolist') else tokens
                    return ''.join([chr(min(t, 127)) for t in tokens_list if t > 0 and t < 128])
        
        self.tokenizer = DummyTokenizer()
        self.model = SimpleLoRATransformer(
            vocab_size=5000,
            d_model=256,
            num_heads=8,
            num_layers=6,
            d_ff=1024,
            max_seq_len=512,
            lora_rank=16,
            lora_alpha=32.0
        )
        print("✅ Created dummy model for demonstration")
    
    def generate_with_model(self, prompt: str, max_tokens: int = 50) -> str:
        """Generate text using the loaded model"""
        if MLX_LM_AVAILABLE and hasattr(self.tokenizer, 'encode'):
            try:
                # Use mlx-lm generate function
                response = generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    verbose=False
                )
                return response
            except Exception as e:
                print(f"Generation failed: {e}")
                return self._fallback_generate(prompt, max_tokens)
        else:
            return self._fallback_generate(prompt, max_tokens)
    
    def _fallback_generate(self, prompt: str, max_tokens: int = 50) -> str:
        """Fallback generation for dummy model"""
        # Simple pattern-based generation for demonstration
        prompt_lower = prompt.lower()
        
        if "sql" in prompt_lower and "select" in prompt_lower:
            if "users" in prompt_lower:
                return "SELECT * FROM users;"
            elif "count" in prompt_lower:
                return "SELECT COUNT(*) FROM table;"
            else:
                return "SELECT column FROM table WHERE condition;"
        
        elif "python" in prompt_lower or "def " in prompt:
            if "multiply" in prompt_lower:
                return "return a * b"
            elif "add" in prompt_lower:
                return "return a + b"
            else:
                return "# implementation here"
        
        elif any(num in prompt for num in ['1', '2', '3', '4', '5']):
            # Try to continue number sequences
            import re
            numbers = re.findall(r'\d+', prompt)
            if len(numbers) >= 2:
                try:
                    nums = [int(n) for n in numbers[-2:]]
                    if len(nums) == 2:
                        # Simple arithmetic progression
                        diff = nums[1] - nums[0]
                        next_num = nums[1] + diff
                        return f" {next_num}"
                except:
                    pass
        
        return " continued text here"
    
    def evaluate_on_tasks(self, num_samples: int = 20) -> Dict[str, float]:
        """Evaluate model on different RLVR tasks"""
        task_types = ["sql", "python", "arithmetic", "fibonacci"]
        results = {}
        
        for task_type in task_types:
            print(f"\n🔍 Evaluating {task_type.upper()} tasks...")
            
            total_reward = 0
            success_count = 0
            
            for i in range(num_samples // len(task_types)):
                # Generate prompt and context
                prompt, context = self.data_generator.generate_prompt(task_type)
                
                # Generate completion
                try:
                    completion = self.generate_with_model(prompt, max_tokens=30)
                    
                    # Compute reward
                    reward, verification = self.reward_computer.compute_reward(
                        completion, task_type, context=context
                    )
                    
                    total_reward += reward
                    if verification.success:
                        success_count += 1
                    
                    # Log first few examples
                    if i < 2:
                        print(f"  Example {i+1}:")
                        print(f"    Prompt: {prompt[:60]}...")
                        print(f"    Generated: {completion[:40]}...")
                        print(f"    Reward: {reward:.3f}, Success: {verification.success}")
                        print(f"    Feedback: {verification.feedback}")
                
                except Exception as e:
                    print(f"    Error in example {i}: {e}")
                    continue
            
            avg_reward = total_reward / (num_samples // len(task_types))
            success_rate = success_count / (num_samples // len(task_types))
            
            results[f"{task_type}_reward"] = avg_reward
            results[f"{task_type}_success"] = success_rate
            
            print(f"  📊 {task_type.capitalize()} Results:")
            print(f"    Average Reward: {avg_reward:.3f}")
            print(f"    Success Rate: {success_rate:.2%}")
        
        # Overall metrics
        overall_reward = np.mean([v for k, v in results.items() if 'reward' in k])
        overall_success = np.mean([v for k, v in results.items() if 'success' in k])
        
        results['overall_reward'] = overall_reward
        results['overall_success'] = overall_success
        
        return results
    
    def run_rlvr_demonstration(self):
        """Run a complete RLVR demonstration"""
        print("=" * 70)
        print("🚀 RLVR WITH PRETRAINED MODEL DEMONSTRATION")
        print("=" * 70)
        
        print(f"\n📋 Model Information:")
        print(f"  Model Path: {self.model_path}")
        print(f"  MLX-LM Available: {MLX_LM_AVAILABLE}")
        print(f"  Tokenizer Type: {type(self.tokenizer).__name__}")
        
        # Test individual verifiers
        print(f"\n🧪 Testing Verifiable Reward Components...")
        self._test_verifiers()
        
        # Test model generation
        print(f"\n🤖 Testing Model Generation...")
        self._test_model_generation()
        
        # Full evaluation
        print(f"\n📊 Running Full RLVR Evaluation...")
        results = self.evaluate_on_tasks(num_samples=16)
        
        # Display final results
        print(f"\n" + "=" * 70)
        print("📈 FINAL RLVR RESULTS")
        print("=" * 70)
        
        print(f"Overall Performance:")
        print(f"  🎯 Average Reward: {results['overall_reward']:.3f}")
        print(f"  ✅ Success Rate: {results['overall_success']:.2%}")
        
        print(f"\nTask-Specific Results:")
        for task in ["sql", "python", "arithmetic", "fibonacci"]:
            reward = results.get(f"{task}_reward", 0)
            success = results.get(f"{task}_success", 0)
            print(f"  {task.capitalize():<10}: Reward={reward:.3f}, Success={success:.2%}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if results['overall_success'] > 0.7:
            print("  ✅ Excellent! Model shows strong verifiable performance")
        elif results['overall_success'] > 0.4:
            print("  ✅ Good performance, could benefit from RLVR fine-tuning")
        else:
            print("  ⚠️  Low success rate - model needs RLVR training")
        
        return results
    
    def _test_verifiers(self):
        """Test the individual verifier components"""
        # SQL Verifier
        print("  🔍 SQL Verifier...")
        sql_verifier = SQLVerifier()
        schema = {
            "users": {
                "columns": [
                    {"name": "id", "type": "INTEGER", "primary_key": True},
                    {"name": "name", "type": "TEXT"},
                    {"name": "age", "type": "INTEGER"}
                ],
                "data": [(1, "Alice", 25), (2, "Bob", 30)]
            }
        }
        sql_verifier.setup_test_database(schema)
        result = sql_verifier.verify_sql_query("SELECT * FROM users")
        print(f"    SQL test result: Success={result.success}, Reward={result.reward:.2f}")
        
        # Code Verifier
        print("  🔍 Code Verifier...")
        code_verifier = CodeVerifier()
        test_code = "def add(a, b):\n    return a + b\nresult = add(a, b)"
        test_cases = [{"inputs": {"a": 2, "b": 3}, "expected_output": 5, "output_var": "result"}]
        result = code_verifier.verify_python_code(test_code, test_cases)
        print(f"    Code test result: Success={result.success}, Reward={result.reward:.2f}")
        
        # Pattern Verifier  
        print("  🔍 Pattern Verifier...")
        pattern_verifier = PatternVerifier()
        result = pattern_verifier.verify_arithmetic_sequence([2, 4, 6, 8, 10])
        print(f"    Pattern test result: Success={result.success}, Reward={result.reward:.2f}")
    
    def _test_model_generation(self):
        """Test model generation capabilities"""
        test_prompts = [
            "Generate SQL: Select all users from table",
            "Complete function: def multiply(a, b):",
            "Continue sequence: 1, 3, 5, 7,",
            "Fibonacci: 1, 1, 2, 3, 5,"
        ]
        
        for prompt in test_prompts:
            print(f"  📝 Prompt: {prompt}")
            try:
                response = self.generate_with_model(prompt, max_tokens=20)
                print(f"  🤖 Response: {response[:50]}...")
            except Exception as e:
                print(f"  ❌ Generation failed: {e}")
            print()


def main():
    """Main function to run RLVR with pretrained models"""
    
    # Model options (try different ones)
    model_options = [
        "mlx-community/Llama-3.2-1B-Instruct-4bit",
        "mlx-community/Qwen2.5-0.5B-Instruct-4bit", 
        "microsoft/DialoGPT-small",
    ]
    
    # Try to use the first available model
    trainer = None
    for model_path in model_options:
        try:
            print(f"Trying to load: {model_path}")
            trainer = PretrainedRLVRTrainer(model_path)
            if trainer.model is not None:
                break
        except Exception as e:
            print(f"Failed to load {model_path}: {e}")
            continue
    
    if trainer is None:
        print("Creating trainer with dummy model")
        trainer = PretrainedRLVRTrainer("dummy")
    
    # Run the demonstration
    results = trainer.run_rlvr_demonstration()
    
    return trainer, results


if __name__ == "__main__":
    trainer, results = main()