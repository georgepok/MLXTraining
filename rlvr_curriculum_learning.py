"""
RLVR Curriculum Learning System
Implements progressive difficulty training to improve SQL and Python generation
"""

import mlx.core as mx
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import random
from dataclasses import dataclass
from collections import deque
import time

from rlvr_improved_training import ImprovedRLVRDataGenerator, ImprovedRLVRRewardComputer


@dataclass
class CurriculumLevel:
    """Represents a curriculum difficulty level"""
    name: str
    difficulty: float  # 0.0 to 1.0
    min_success_rate: float  # Required success rate to advance
    max_attempts: int  # Max attempts before moving on
    task_weights: Dict[str, float]  # Task type probabilities
    constraints: Dict[str, Any]  # Task-specific constraints


class AdvancedCurriculumScheduler:
    """Advanced curriculum learning scheduler for RLVR"""
    
    def __init__(self):
        self.levels = self._create_curriculum_levels()
        self.current_level = 0
        self.level_attempts = 0
        self.level_successes = 0
        self.success_history = deque(maxlen=50)  # Track recent success rate
        self.level_start_time = time.time()
        
    def _create_curriculum_levels(self) -> List[CurriculumLevel]:
        """Create progressive curriculum levels"""
        return [
            # Level 0: Basics
            CurriculumLevel(
                name="Basic Queries",
                difficulty=0.1,
                min_success_rate=0.7,
                max_attempts=100,
                task_weights={"sql": 0.4, "python": 0.4, "arithmetic": 0.2, "fibonacci": 0.0},
                constraints={
                    "sql_complexity": "basic_select",
                    "python_complexity": "basic_math",
                    "max_sql_conditions": 0,
                    "max_python_lines": 2
                }
            ),
            
            # Level 1: Simple Filtering
            CurriculumLevel(
                name="Simple Filtering",
                difficulty=0.3,
                min_success_rate=0.6,
                max_attempts=150,
                task_weights={"sql": 0.5, "python": 0.3, "arithmetic": 0.15, "fibonacci": 0.05},
                constraints={
                    "sql_complexity": "filtered_select",
                    "python_complexity": "basic_math",
                    "max_sql_conditions": 1,
                    "max_python_lines": 3
                }
            ),
            
            # Level 2: Aggregations and String Operations
            CurriculumLevel(
                name="Aggregations & Strings",
                difficulty=0.5,
                min_success_rate=0.5,
                max_attempts=200,
                task_weights={"sql": 0.4, "python": 0.4, "arithmetic": 0.1, "fibonacci": 0.1},
                constraints={
                    "sql_complexity": "aggregation",
                    "python_complexity": "string_operations",
                    "max_sql_conditions": 2,
                    "max_python_lines": 4
                }
            ),
            
            # Level 3: Complex Queries and List Operations
            CurriculumLevel(
                name="Complex Operations",
                difficulty=0.7,
                min_success_rate=0.45,
                max_attempts=250,
                task_weights={"sql": 0.35, "python": 0.45, "arithmetic": 0.1, "fibonacci": 0.1},
                constraints={
                    "sql_complexity": "grouping",
                    "python_complexity": "list_operations",
                    "max_sql_conditions": 3,
                    "max_python_lines": 6
                }
            ),
            
            # Level 4: Advanced Features
            CurriculumLevel(
                name="Advanced Features",
                difficulty=0.9,
                min_success_rate=0.4,
                max_attempts=300,
                task_weights={"sql": 0.3, "python": 0.5, "arithmetic": 0.1, "fibonacci": 0.1},
                constraints={
                    "sql_complexity": "complex",
                    "python_complexity": "algorithms",
                    "max_sql_conditions": 5,
                    "max_python_lines": 10
                }
            ),
            
            # Level 5: Expert Level
            CurriculumLevel(
                name="Expert Level",
                difficulty=1.0,
                min_success_rate=0.35,
                max_attempts=500,
                task_weights={"sql": 0.25, "python": 0.5, "arithmetic": 0.125, "fibonacci": 0.125},
                constraints={
                    "sql_complexity": "mixed",
                    "python_complexity": "mixed",
                    "max_sql_conditions": 10,
                    "max_python_lines": 15
                }
            )
        ]
    
    def get_current_level(self) -> CurriculumLevel:
        """Get current curriculum level"""
        return self.levels[min(self.current_level, len(self.levels) - 1)]
    
    def update_performance(self, success: bool, task_type: str, reward: float):
        """Update curriculum based on performance"""
        self.level_attempts += 1
        if success:
            self.level_successes += 1
        
        self.success_history.append(success)
        
        # Check if we should advance to next level
        current_level = self.get_current_level()
        
        if len(self.success_history) >= 20:  # Need enough data
            recent_success_rate = sum(self.success_history) / len(self.success_history)
            
            # Advance if success rate is good enough
            if recent_success_rate >= current_level.min_success_rate:
                if self.current_level < len(self.levels) - 1:
                    self._advance_level()
            
            # Also advance if max attempts reached (with minimum success)
            elif (self.level_attempts >= current_level.max_attempts and 
                  recent_success_rate >= 0.2):  # At least some success
                if self.current_level < len(self.levels) - 1:
                    self._advance_level()
    
    def _advance_level(self):
        """Advance to next curriculum level"""
        old_level = self.current_level
        self.current_level += 1
        self.level_attempts = 0
        self.level_successes = 0
        self.success_history.clear()
        self.level_start_time = time.time()
        
        print(f"\n🎓 Curriculum Advanced: Level {old_level} → Level {self.current_level}")
        print(f"   New Level: {self.get_current_level().name}")
        print(f"   Difficulty: {self.get_current_level().difficulty:.2f}")
    
    def get_task_distribution(self) -> Dict[str, float]:
        """Get task distribution for current level"""
        return self.get_current_level().task_weights
    
    def get_constraints(self) -> Dict[str, Any]:
        """Get constraints for current level"""
        return self.get_current_level().constraints
    
    def get_stats(self) -> Dict[str, Any]:
        """Get curriculum statistics"""
        current_level = self.get_current_level()
        recent_success_rate = (sum(self.success_history) / len(self.success_history) 
                              if self.success_history else 0)
        
        return {
            "current_level": self.current_level,
            "level_name": current_level.name,
            "difficulty": current_level.difficulty,
            "level_attempts": self.level_attempts,
            "level_successes": self.level_successes,
            "recent_success_rate": recent_success_rate,
            "time_in_level": time.time() - self.level_start_time,
            "total_levels": len(self.levels)
        }


class CurriculumAwareDataGenerator:
    """Data generator that adapts to curriculum level"""
    
    def __init__(self, tokenizer=None, max_seq_len: int = 512):
        self.base_generator = ImprovedRLVRDataGenerator(tokenizer, max_seq_len)
        self.curriculum = AdvancedCurriculumScheduler()
    
    def generate_prompt(self, task_type: str = None) -> Tuple[str, Dict]:
        """Generate prompt adapted to current curriculum level"""
        # Get current curriculum constraints
        constraints = self.curriculum.get_constraints()
        
        # Select task type based on curriculum weights if not specified
        if task_type is None:
            task_weights = self.curriculum.get_task_distribution()
            task_type = np.random.choice(
                list(task_weights.keys()),
                p=list(task_weights.values())
            )
        
        # Generate base prompt
        prompt, context = self.base_generator.generate_prompt(task_type)
        
        # Adapt prompt based on curriculum constraints
        if task_type == "sql":
            prompt, context = self._adapt_sql_prompt(prompt, context, constraints)
        elif task_type == "python":
            prompt, context = self._adapt_python_prompt(prompt, context, constraints)
        
        # Add curriculum level to context
        context["curriculum_level"] = self.curriculum.current_level
        context["difficulty"] = self.curriculum.get_current_level().difficulty
        
        return prompt, context
    
    def _adapt_sql_prompt(self, prompt: str, context: Dict, constraints: Dict) -> Tuple[str, Dict]:
        """Adapt SQL prompt based on curriculum constraints"""
        sql_complexity = constraints.get("sql_complexity", "mixed")
        max_conditions = constraints.get("max_sql_conditions", 10)
        
        if sql_complexity != "mixed":
            # Force specific complexity level
            if sql_complexity == "basic_select":
                # Simplify to basic SELECT
                table_name = context.get("table", "users")
                context["question"] = f"Select all {table_name}"
                context["expected_query"] = f"SELECT * FROM {table_name};"
                
                prompt = f"""You are an expert SQL developer. Write a SQL query for the following request.

Database Schema:
{self._format_simple_schema(context)}

Task: Select all {table_name}

Write only the SQL query without any explanation:"""
        
        return prompt, context
    
    def _adapt_python_prompt(self, prompt: str, context: Dict, constraints: Dict) -> Tuple[str, Dict]:
        """Adapt Python prompt based on curriculum constraints"""
        python_complexity = constraints.get("python_complexity", "mixed")
        max_lines = constraints.get("max_python_lines", 15)
        
        if python_complexity != "mixed":
            # Filter to specific complexity
            if python_complexity == "basic_math":
                # Force basic math operations
                basic_functions = ["add", "multiply", "subtract"]
                if any(func in context.get("signature", "") for func in basic_functions):
                    # Keep as is
                    pass
                else:
                    # Replace with basic math
                    signature = "def add(a, b):"
                    description = "Add two numbers and return the result"
                    test_cases = [
                        {"inputs": {"a": 5, "b": 3}, "expected_output": 8, "output_var": "result"}
                    ]
                    expected_code = "return a + b"
                    
                    context.update({
                        "signature": signature,
                        "description": description,
                        "test_cases": test_cases,
                        "expected_code": expected_code
                    })
                    
                    prompt = f"""You are an expert Python programmer. Complete the following function.

Function: {signature}
Task: {description}

Test Cases:
  Test 1: add(a=5, b=3) should return 8

Complete the function implementation:
{signature}
    # Your code here"""
        
        return prompt, context
    
    def _format_simple_schema(self, context: Dict) -> str:
        """Format simplified schema for basic levels"""
        schema = context.get("schema", {})
        if not schema:
            return "Table: users\nColumns: id, name, email"
        
        table_name = list(schema.keys())[0]
        table_info = schema[table_name]
        columns = [col["name"] for col in table_info.get("columns", [])]
        
        return f"Table: {table_name}\nColumns: {', '.join(columns)}"
    
    def update_performance(self, success: bool, task_type: str, reward: float):
        """Update curriculum based on performance"""
        self.curriculum.update_performance(success, task_type, reward)
    
    def get_curriculum_stats(self) -> Dict[str, Any]:
        """Get curriculum statistics"""
        return self.curriculum.get_stats()


class CurriculumTrainingLoop:
    """Training loop with curriculum learning"""
    
    def __init__(self, model, tokenizer=None):
        self.model = model
        self.data_generator = CurriculumAwareDataGenerator(tokenizer)
        self.reward_computer = ImprovedRLVRRewardComputer()
        self.stats = {
            "total_samples": 0,
            "total_successes": 0,
            "level_transitions": [],
            "task_performance": {"sql": [], "python": [], "arithmetic": [], "fibonacci": []}
        }
    
    def train_step(self) -> Dict[str, Any]:
        """Single training step with curriculum learning"""
        # Generate prompt based on curriculum
        prompt, context = self.data_generator.generate_prompt()
        task_type = self._infer_task_type(context)
        
        # Generate completion (simplified for demo)
        completion = self._generate_completion(prompt, context, task_type)
        
        # Compute reward
        reward, verification = self.reward_computer.compute_reward(
            completion, task_type, context
        )
        
        # Update curriculum
        self.data_generator.update_performance(verification.success, task_type, reward)
        
        # Update stats
        self.stats["total_samples"] += 1
        if verification.success:
            self.stats["total_successes"] += 1
        
        self.stats["task_performance"][task_type].append({
            "reward": reward,
            "success": verification.success,
            "curriculum_level": context.get("curriculum_level", 0)
        })
        
        # Check for level transitions
        curriculum_stats = self.data_generator.get_curriculum_stats()
        if (len(self.stats["level_transitions"]) == 0 or 
            self.stats["level_transitions"][-1]["to_level"] != curriculum_stats["current_level"]):
            self.stats["level_transitions"].append({
                "step": self.stats["total_samples"],
                "to_level": curriculum_stats["current_level"],
                "level_name": curriculum_stats["level_name"]
            })
        
        return {
            "reward": reward,
            "success": verification.success,
            "task_type": task_type,
            "completion": completion[:50] + "..." if len(completion) > 50 else completion,
            "curriculum_level": curriculum_stats["current_level"],
            "level_name": curriculum_stats["level_name"],
            "curriculum_stats": curriculum_stats
        }
    
    def _infer_task_type(self, context: Dict) -> str:
        """Infer task type from context"""
        if "schema" in context or "expected_query" in context:
            return "sql"
        elif "signature" in context or "test_cases" in context:
            return "python"
        elif "sequence" in context and "fibonacci" in str(context):
            return "fibonacci"
        elif "sequence" in context or "step" in context:
            return "arithmetic"
        else:
            return "sql"  # Default
    
    def _generate_completion(self, prompt: str, context: Dict, task_type: str) -> str:
        """Generate completion (simplified for demo)"""
        # This would be replaced with actual model generation
        if task_type == "sql":
            if "expected_query" in context:
                return context["expected_query"]
            return "SELECT * FROM users;"
        elif task_type == "python":
            if "expected_code" in context:
                signature = context.get("signature", "def func():")
                return f"{signature}\n    {context['expected_code']}"
            return "return a + b"
        elif task_type in ["arithmetic", "fibonacci"]:
            if "next_number" in context:
                return str(context["next_number"])
            return "42"
        else:
            return "result"
    
    def train(self, num_steps: int, log_interval: int = 50):
        """Run curriculum training"""
        print("🎓 Starting Curriculum-Based RLVR Training")
        print("=" * 60)
        
        for step in range(num_steps):
            metrics = self.train_step()
            
            if (step + 1) % log_interval == 0:
                self._log_progress(step + 1, metrics)
        
        self._print_final_stats()
    
    def _log_progress(self, step: int, metrics: Dict):
        """Log training progress"""
        overall_success = (self.stats["total_successes"] / self.stats["total_samples"] 
                          if self.stats["total_samples"] > 0 else 0)
        
        print(f"\nStep {step}:")
        print(f"  📊 Overall Success: {overall_success:.1%}")
        print(f"  🎯 Current Task: {metrics['task_type']} | Reward: {metrics['reward']:.2f}")
        print(f"  🎓 Level: {metrics['curriculum_level']} - {metrics['level_name']}")
        print(f"  💬 Completion: {metrics['completion']}")
        
        # Task-specific success rates
        for task_type, performances in self.stats["task_performance"].items():
            if performances:
                recent_performances = performances[-20:]  # Last 20 attempts
                success_rate = sum(p["success"] for p in recent_performances) / len(recent_performances)
                avg_reward = sum(p["reward"] for p in recent_performances) / len(recent_performances)
                print(f"  {task_type.upper()}: Success={success_rate:.1%}, Reward={avg_reward:.2f}")
    
    def _print_final_stats(self):
        """Print final training statistics"""
        print("\n" + "=" * 60)
        print("🏁 CURRICULUM TRAINING COMPLETE")
        print("=" * 60)
        
        overall_success = (self.stats["total_successes"] / self.stats["total_samples"] 
                          if self.stats["total_samples"] > 0 else 0)
        
        print(f"📈 Overall Results:")
        print(f"  Total Samples: {self.stats['total_samples']}")
        print(f"  Total Successes: {self.stats['total_successes']}")
        print(f"  Overall Success Rate: {overall_success:.1%}")
        
        print(f"\n🎓 Curriculum Progress:")
        final_stats = self.data_generator.get_curriculum_stats()
        print(f"  Final Level: {final_stats['current_level']} - {final_stats['level_name']}")
        print(f"  Level Transitions: {len(self.stats['level_transitions'])}")
        
        for transition in self.stats["level_transitions"]:
            print(f"    Step {transition['step']}: Advanced to Level {transition['to_level']} - {transition['level_name']}")
        
        print(f"\n📊 Task Performance Summary:")
        for task_type, performances in self.stats["task_performance"].items():
            if performances:
                successes = sum(p["success"] for p in performances)
                success_rate = successes / len(performances)
                avg_reward = sum(p["reward"] for p in performances) / len(performances)
                print(f"  {task_type.upper()}: {successes}/{len(performances)} ({success_rate:.1%}) - Avg Reward: {avg_reward:.2f}")


def test_curriculum_learning():
    """Test curriculum learning system"""
    print("Testing Curriculum Learning System")
    print("=" * 50)
    
    # Mock model for testing
    class MockModel:
        def __call__(self, *args, **kwargs):
            return mx.array([[1, 2, 3]])
    
    model = MockModel()
    
    # Create curriculum training loop
    trainer = CurriculumTrainingLoop(model)
    
    # Run short training
    trainer.train(num_steps=200, log_interval=40)


if __name__ == "__main__":
    test_curriculum_learning()