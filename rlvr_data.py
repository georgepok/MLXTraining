"""
RLVR Data Module for Experience Buffer and Trajectory Sampling
Handles data generation, storage, and sampling for RL training
"""

import mlx.core as mx
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import random
from collections import deque
import json


@dataclass
class Trajectory:
    """Single trajectory from policy rollout"""
    states: mx.array  # Input token ids [seq_len]
    actions: mx.array  # Generated token ids [seq_len]
    rewards: float  # Scalar reward
    log_probs: Optional[mx.array] = None  # Log probabilities [seq_len]
    values: Optional[mx.array] = None  # Value estimates
    text: Optional[str] = None  # Generated text
    task_type: Optional[str] = None  # Task identifier
    context: Optional[Dict] = None  # Task context
    metadata: Dict = field(default_factory=dict)  # Additional metadata


class ExperienceBuffer:
    """Buffer for storing and sampling trajectories"""
    
    def __init__(self, capacity: int = 10000, prioritized: bool = False):
        """
        Initialize experience buffer
        
        Args:
            capacity: Maximum number of trajectories to store
            prioritized: Whether to use prioritized experience replay
        """
        self.capacity = capacity
        self.prioritized = prioritized
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity) if prioritized else None
        self.total_added = 0
    
    def add(self, trajectory: Trajectory, priority: Optional[float] = None):
        """Add trajectory to buffer"""
        self.buffer.append(trajectory)
        if self.prioritized and priority is not None:
            self.priorities.append(priority)
        elif self.prioritized:
            # Default priority based on reward
            self.priorities.append(abs(trajectory.rewards) + 1e-6)
        self.total_added += 1
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Trajectory], Optional[mx.array]]:
        """
        Sample batch of trajectories
        
        Args:
            batch_size: Number of trajectories to sample
            beta: Importance sampling weight for prioritized replay
        
        Returns:
            Tuple of (trajectories, importance_weights)
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        if self.prioritized and self.priorities:
            # Prioritized sampling
            priorities = np.array(self.priorities)
            probs = priorities / priorities.sum()
            indices = np.random.choice(len(self.buffer), batch_size, p=probs)
            
            # Importance sampling weights
            weights = (len(self.buffer) * probs[indices]) ** (-beta)
            weights = weights / weights.max()
            weights = mx.array(weights)
            
            trajectories = [self.buffer[i] for i in indices]
            return trajectories, weights
        else:
            # Uniform sampling
            trajectories = random.sample(self.buffer, batch_size)
            return trajectories, None
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for prioritized replay"""
        if self.prioritized and self.priorities:
            for idx, priority in zip(indices, priorities):
                if 0 <= idx < len(self.priorities):
                    self.priorities[idx] = priority + 1e-6
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
        if self.priorities:
            self.priorities.clear()
    
    def __len__(self):
        return len(self.buffer)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get buffer statistics"""
        if not self.buffer:
            return {}
        
        rewards = [t.rewards for t in self.buffer]
        return {
            "size": len(self.buffer),
            "total_added": self.total_added,
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "max_reward": np.max(rewards),
            "min_reward": np.min(rewards)
        }


class RLVRDataGenerator:
    """Generate prompts and contexts for RLVR training"""
    
    def __init__(self, 
                 tokenizer=None,
                 max_seq_len: int = 512,
                 task_configs: Optional[Dict] = None):
        """
        Initialize data generator
        
        Args:
            tokenizer: Tokenizer for text encoding
            max_seq_len: Maximum sequence length
            task_configs: Configuration for different task types
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.task_configs = task_configs or self._get_default_task_configs()
        
        # Special tokens
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.sep_token_id = 3
    
    def _get_default_task_configs(self) -> Dict:
        """Get default task configurations"""
        return {
            "sql": {
                "prompt_template": "Generate SQL query for: {question}\nTable: {table}\nSQL:",
                "max_gen_len": 100,
                "temperature": 0.7,
                "schemas": self._get_sample_sql_schemas()
            },
            "python": {
                "prompt_template": "Complete the Python function:\n{code_context}\n",
                "max_gen_len": 150,
                "temperature": 0.8,
                "test_cases": self._get_sample_python_tests()
            },
            "arithmetic": {
                "prompt_template": "Continue the arithmetic sequence: {sequence}",
                "max_gen_len": 50,
                "temperature": 0.5
            },
            "fibonacci": {
                "prompt_template": "Continue the Fibonacci sequence: {sequence}",
                "max_gen_len": 50,
                "temperature": 0.5
            }
        }
    
    def _get_sample_sql_schemas(self) -> List[Dict]:
        """Get sample SQL schemas for training"""
        return [
            {
                "users": {
                    "columns": [
                        {"name": "id", "type": "INTEGER", "primary_key": True},
                        {"name": "name", "type": "TEXT"},
                        {"name": "email", "type": "TEXT"},
                        {"name": "age", "type": "INTEGER"}
                    ],
                    "data": [
                        (1, "Alice", "alice@example.com", 25),
                        (2, "Bob", "bob@example.com", 30),
                        (3, "Charlie", "charlie@example.com", 35)
                    ]
                }
            },
            {
                "products": {
                    "columns": [
                        {"name": "id", "type": "INTEGER", "primary_key": True},
                        {"name": "name", "type": "TEXT"},
                        {"name": "price", "type": "REAL"},
                        {"name": "category", "type": "TEXT"}
                    ],
                    "data": [
                        (1, "Laptop", 999.99, "Electronics"),
                        (2, "Phone", 699.99, "Electronics"),
                        (3, "Desk", 299.99, "Furniture")
                    ]
                }
            }
        ]
    
    def _get_sample_python_tests(self) -> List[Dict]:
        """Get sample Python test cases"""
        return [
            {
                "code_context": "def add(a, b):",
                "test_cases": [
                    {"inputs": {"a": 1, "b": 2}, "expected_output": 3},
                    {"inputs": {"a": -1, "b": 1}, "expected_output": 0}
                ]
            },
            {
                "code_context": "def factorial(n):",
                "test_cases": [
                    {"inputs": {"n": 5}, "expected_output": 120},
                    {"inputs": {"n": 0}, "expected_output": 1}
                ]
            }
        ]
    
    def generate_prompt(self, task_type: str) -> Tuple[str, Dict]:
        """
        Generate a prompt and context for a given task type
        
        Returns:
            Tuple of (prompt_text, context_dict)
        """
        config = self.task_configs.get(task_type, {})
        
        if task_type == "sql":
            schema = random.choice(config["schemas"])
            table_name = list(schema.keys())[0]
            columns = [col["name"] for col in schema[table_name]["columns"]]
            
            questions = [
                f"Select all {table_name}",
                f"Count total {table_name}",
                f"Select {random.choice(columns)} from {table_name}",
                f"Select {table_name} where {random.choice(columns)} > 10"
            ]
            
            question = random.choice(questions)
            prompt = config["prompt_template"].format(
                question=question,
                table=f"{table_name}({', '.join(columns)})"
            )
            
            context = {
                "schema": schema,
                "question": question,
                "table": table_name
            }
            
        elif task_type == "python":
            test_config = random.choice(config["test_cases"])
            prompt = config["prompt_template"].format(
                code_context=test_config["code_context"]
            )
            context = test_config
            
        elif task_type == "arithmetic":
            start = random.randint(1, 10)
            step = random.randint(1, 5)
            length = random.randint(3, 5)
            sequence = [start + i * step for i in range(length)]
            
            prompt = config["prompt_template"].format(
                sequence=", ".join(map(str, sequence))
            )
            context = {"start": start, "step": step, "difference": step}
            
        elif task_type == "fibonacci":
            sequence = [1, 1]
            for i in range(random.randint(2, 4)):
                sequence.append(sequence[-1] + sequence[-2])
            
            prompt = config["prompt_template"].format(
                sequence=", ".join(map(str, sequence))
            )
            context = {"sequence": sequence}
            
        else:
            # Default text generation
            prompt = "Generate text:"
            context = {}
        
        return prompt, context
    
    def tokenize(self, text: str) -> mx.array:
        """Tokenize text to token ids"""
        if self.tokenizer:
            tokens = self.tokenizer.encode(text)
            # Truncate if needed
            if len(tokens) > self.max_seq_len:
                tokens = tokens[:self.max_seq_len]
            # Pad if needed
            while len(tokens) < self.max_seq_len:
                tokens.append(self.pad_token_id)
            return mx.array(tokens)
        else:
            # Simple character-level tokenization as fallback
            tokens = [ord(c) % 1000 for c in text[:self.max_seq_len]]
            while len(tokens) < self.max_seq_len:
                tokens.append(self.pad_token_id)
            return mx.array(tokens)
    
    def decode(self, token_ids: mx.array) -> str:
        """Decode token ids to text"""
        if self.tokenizer:
            return self.tokenizer.decode(token_ids.tolist())
        else:
            # Simple character-level decoding as fallback
            chars = []
            for token_id in token_ids.tolist():
                if token_id == self.pad_token_id:
                    break
                if token_id < 128:
                    chars.append(chr(token_id))
            return ''.join(chars)
    
    def generate_rollouts(self, 
                         model,
                         batch_size: int,
                         task_distribution: Optional[Dict[str, float]] = None) -> Tuple[Dict, List[str], List[Dict]]:
        """
        Generate rollouts using the model
        
        Args:
            model: Policy model for generation
            batch_size: Number of rollouts to generate
            task_distribution: Probability distribution over task types
        
        Returns:
            Tuple of (trajectories_dict, task_types, contexts)
        """
        if task_distribution is None:
            task_distribution = {
                "sql": 0.3,
                "python": 0.3,
                "arithmetic": 0.2,
                "fibonacci": 0.2
            }
        
        # Normalize distribution
        total = sum(task_distribution.values())
        task_distribution = {k: v/total for k, v in task_distribution.items()}
        
        # Sample task types
        task_types = np.random.choice(
            list(task_distribution.keys()),
            size=batch_size,
            p=list(task_distribution.values())
        ).tolist()
        
        # Generate prompts and contexts
        prompts = []
        contexts = []
        for task_type in task_types:
            prompt, context = self.generate_prompt(task_type)
            prompts.append(prompt)
            contexts.append(context)
        
        # Tokenize prompts
        input_ids = mx.stack([self.tokenize(p) for p in prompts])
        
        # Generate completions
        # In MLX, gradients are not computed by default, so no need for no_grad()
        generated_ids = self._generate(model, input_ids, max_new_tokens=50)
        
        # Decode generated text
        generated_texts = [self.decode(ids) for ids in generated_ids]
        
        # Create trajectories dictionary
        trajectories = {
            "states": input_ids,
            "actions": generated_ids,
            "texts": generated_texts,
            "masks": (input_ids != self.pad_token_id).astype(mx.float32)
        }
        
        return trajectories, task_types, contexts
    
    def _generate(self, model, input_ids: mx.array, max_new_tokens: int = 50) -> mx.array:
        """Simple generation function"""
        batch_size, seq_len = input_ids.shape
        generated = input_ids.tolist()
        
        for _ in range(max_new_tokens):
            # Get model predictions
            current_ids = mx.array(generated)
            logits = model(current_ids)
            
            # Get next token (greedy for simplicity)
            next_tokens = mx.argmax(logits[:, -1, :], axis=-1)
            
            # Append to generated
            for i, next_token in enumerate(next_tokens.tolist()):
                if len(generated[i]) < self.max_seq_len:
                    generated[i].append(next_token)
                    if next_token == self.eos_token_id:
                        break
        
        # Pad sequences to same length
        max_len = max(len(seq) for seq in generated)
        for i in range(batch_size):
            while len(generated[i]) < max_len:
                generated[i].append(self.pad_token_id)
        
        return mx.array(generated)


class RLVRDataset:
    """Dataset class for RLVR training"""
    
    def __init__(self, 
                 data_path: Optional[str] = None,
                 task_type: str = "mixed"):
        """
        Initialize dataset
        
        Args:
            data_path: Path to dataset file
            task_type: Type of task or "mixed"
        """
        self.data_path = data_path
        self.task_type = task_type
        self.data = []
        
        if data_path:
            self.load_data()
    
    def load_data(self):
        """Load data from file"""
        if self.data_path.endswith('.json'):
            with open(self.data_path, 'r') as f:
                self.data = json.load(f)
        elif self.data_path.endswith('.jsonl'):
            with open(self.data_path, 'r') as f:
                self.data = [json.loads(line) for line in f]
        else:
            raise ValueError(f"Unsupported file format: {self.data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def get_batch(self, batch_size: int) -> List[Dict]:
        """Get random batch from dataset"""
        if len(self.data) < batch_size:
            # Sample with replacement if dataset is small
            indices = np.random.choice(len(self.data), batch_size, replace=True)
        else:
            indices = np.random.choice(len(self.data), batch_size, replace=False)
        
        return [self.data[i] for i in indices]


class CurriculumScheduler:
    """Curriculum learning scheduler for RLVR"""
    
    def __init__(self, 
                 initial_difficulty: float = 0.1,
                 max_difficulty: float = 1.0,
                 warmup_steps: int = 100,
                 increase_rate: float = 0.001):
        """
        Initialize curriculum scheduler
        
        Args:
            initial_difficulty: Starting difficulty level
            max_difficulty: Maximum difficulty level
            warmup_steps: Steps before increasing difficulty
            increase_rate: Rate of difficulty increase
        """
        self.initial_difficulty = initial_difficulty
        self.max_difficulty = max_difficulty
        self.warmup_steps = warmup_steps
        self.increase_rate = increase_rate
        self.current_step = 0
        self.current_difficulty = initial_difficulty
        
        # Performance tracking
        self.success_rate_window = deque(maxlen=100)
    
    def get_difficulty(self) -> float:
        """Get current difficulty level"""
        return self.current_difficulty
    
    def update(self, success_rate: float):
        """Update difficulty based on performance"""
        self.current_step += 1
        self.success_rate_window.append(success_rate)
        
        if self.current_step < self.warmup_steps:
            return
        
        # Adjust difficulty based on success rate
        avg_success = np.mean(self.success_rate_window) if self.success_rate_window else 0
        
        if avg_success > 0.8:  # Too easy
            self.current_difficulty = min(
                self.max_difficulty,
                self.current_difficulty + self.increase_rate
            )
        elif avg_success < 0.3:  # Too hard
            self.current_difficulty = max(
                self.initial_difficulty,
                self.current_difficulty - self.increase_rate * 0.5
            )
    
    def get_task_config(self, base_config: Dict) -> Dict:
        """Modify task config based on current difficulty"""
        config = base_config.copy()
        
        # Adjust task parameters based on difficulty
        if "max_gen_len" in config:
            config["max_gen_len"] = int(
                config["max_gen_len"] * (0.5 + 0.5 * self.current_difficulty)
            )
        
        if "temperature" in config:
            # Higher difficulty = lower temperature (more deterministic)
            config["temperature"] = config["temperature"] * (1.5 - 0.5 * self.current_difficulty)
        
        return config