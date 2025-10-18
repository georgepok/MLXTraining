"""
RLVR (Reinforcement Learning from Verifiable Rewards) Core Module
Implements verifiable reward functions and correctness checking for MLX
"""

import mlx.core as mx
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import sqlite3
import tempfile
import subprocess
import json
import re
from enum import Enum


class RewardType(Enum):
    """Types of verifiable rewards"""
    EXECUTION_SUCCESS = "execution_success"
    SYNTAX_VALID = "syntax_valid"
    OUTPUT_MATCH = "output_match"
    PERFORMANCE = "performance"
    CORRECTNESS = "correctness"


@dataclass
class VerificationResult:
    """Result of verification for a generated output"""
    success: bool
    reward: float
    feedback: str
    details: Dict[str, Any]
    execution_output: Optional[str] = None
    error_message: Optional[str] = None


class SQLVerifier:
    """Verifies SQL queries through execution and result checking"""
    
    def __init__(self, database_path: Optional[str] = None):
        self.database_path = database_path or ":memory:"
        self.connection = None
        
    def setup_test_database(self, schema: Dict[str, Dict]):
        """Setup a test database with given schema"""
        self.connection = sqlite3.connect(self.database_path)
        cursor = self.connection.cursor()
        
        for table_name, table_config in schema.items():
            # Create table
            col_definitions = []
            columns = table_config.get('columns', [])
            
            for col in columns:
                col_def = f"{col['name']} {col['type']}"
                if col.get('primary_key'):
                    col_def += " PRIMARY KEY"
                col_definitions.append(col_def)
            
            create_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(col_definitions)})"
            cursor.execute(create_query)
            
            # Insert sample data if provided
            if 'data' in table_config:
                for row in table_config['data']:
                    placeholders = ', '.join(['?' for _ in row])
                    insert_query = f"INSERT INTO {table_name} VALUES ({placeholders})"
                    cursor.execute(insert_query, row)
        
        self.connection.commit()
    
    def verify_sql_query(self, query: str, expected_result: Optional[Any] = None) -> VerificationResult:
        """Verify a SQL query by executing it"""
        if not self.connection:
            self.connection = sqlite3.connect(self.database_path)
        
        # Check syntax validity
        try:
            # Parse query for basic syntax check
            query_clean = query.strip().rstrip(';')
            if not query_clean:
                return VerificationResult(
                    success=False,
                    reward=-1.0,
                    feedback="Empty query",
                    details={"error": "Query is empty"}
                )
            
            # Execute query
            cursor = self.connection.cursor()
            cursor.execute(query_clean)
            
            # Get results
            if query_clean.upper().startswith('SELECT'):
                results = cursor.fetchall()
                execution_output = str(results)
            else:
                self.connection.commit()
                results = cursor.rowcount
                execution_output = f"Affected rows: {results}"
            
            # Calculate reward
            base_reward = 1.0  # Successful execution
            
            # Check against expected result if provided
            if expected_result is not None:
                if results == expected_result:
                    reward = 2.0  # Perfect match
                    feedback = "Query executed successfully with correct output"
                else:
                    reward = 0.5  # Executed but wrong result
                    feedback = "Query executed but output doesn't match expected"
            else:
                reward = base_reward
                feedback = "Query executed successfully"
            
            return VerificationResult(
                success=True,
                reward=reward,
                feedback=feedback,
                details={
                    "query": query,
                    "results": results,
                    "row_count": len(results) if isinstance(results, list) else results
                },
                execution_output=execution_output
            )
            
        except sqlite3.Error as e:
            return VerificationResult(
                success=False,
                reward=-0.5,
                feedback=f"SQL execution error: {str(e)}",
                details={"error": str(e), "query": query},
                error_message=str(e)
            )
        except Exception as e:
            return VerificationResult(
                success=False,
                reward=-1.0,
                feedback=f"Unexpected error: {str(e)}",
                details={"error": str(e), "query": query},
                error_message=str(e)
            )
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()


class CodeVerifier:
    """Verifies code through execution and output checking"""
    
    @staticmethod
    def verify_python_code(code: str, test_cases: List[Dict] = None, 
                          timeout: int = 5) -> VerificationResult:
        """Verify Python code by executing it with test cases"""
        
        # Basic syntax check
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            return VerificationResult(
                success=False,
                reward=-1.0,
                feedback=f"Syntax error: {e}",
                details={"error": str(e), "line": e.lineno},
                error_message=str(e)
            )
        
        # Execute code with test cases
        if test_cases:
            total_reward = 0.0
            passed_tests = 0
            failed_tests = []
            
            for i, test in enumerate(test_cases):
                try:
                    # Create execution environment
                    exec_globals = {}
                    exec_locals = {}
                    
                    # Add input variables
                    if 'inputs' in test:
                        exec_locals.update(test['inputs'])
                    
                    # Execute code
                    exec(code, exec_globals, exec_locals)
                    
                    # Check output
                    if 'expected_output' in test:
                        if 'output_var' in test:
                            actual = exec_locals.get(test['output_var'])
                        else:
                            # Try to get return value of main function if exists
                            actual = exec_locals.get('result', None)
                        
                        if actual == test['expected_output']:
                            total_reward += 1.0
                            passed_tests += 1
                        else:
                            total_reward -= 0.5
                            failed_tests.append({
                                "test_id": i,
                                "expected": test['expected_output'],
                                "actual": actual
                            })
                    else:
                        # No expected output, just check execution
                        total_reward += 0.5
                        passed_tests += 1
                        
                except Exception as e:
                    total_reward -= 1.0
                    failed_tests.append({
                        "test_id": i,
                        "error": str(e)
                    })
            
            # Calculate final reward
            avg_reward = total_reward / len(test_cases) if test_cases else 0
            success = passed_tests > len(test_cases) / 2
            
            return VerificationResult(
                success=success,
                reward=avg_reward,
                feedback=f"Passed {passed_tests}/{len(test_cases)} test cases",
                details={
                    "passed": passed_tests,
                    "total": len(test_cases),
                    "failed_tests": failed_tests
                }
            )
        else:
            # Just execute without test cases
            try:
                exec_globals = {}
                exec(code, exec_globals)
                
                return VerificationResult(
                    success=True,
                    reward=0.5,
                    feedback="Code executed without errors",
                    details={"execution": "successful"}
                )
            except Exception as e:
                return VerificationResult(
                    success=False,
                    reward=-0.5,
                    feedback=f"Runtime error: {e}",
                    details={"error": str(e)},
                    error_message=str(e)
                )


class PatternVerifier:
    """Verifies pattern completion and sequence generation"""
    
    @staticmethod
    def verify_arithmetic_sequence(sequence: List[int], expected_diff: Optional[int] = None) -> VerificationResult:
        """Verify if sequence follows arithmetic progression"""
        if len(sequence) < 3:
            return VerificationResult(
                success=False,
                reward=-0.5,
                feedback="Sequence too short to verify",
                details={"length": len(sequence)}
            )
        
        # Calculate differences
        diffs = [sequence[i+1] - sequence[i] for i in range(len(sequence)-1)]
        
        # Check consistency
        if len(set(diffs)) == 1:
            actual_diff = diffs[0]
            if expected_diff is not None:
                if actual_diff == expected_diff:
                    reward = 2.0
                    feedback = f"Perfect arithmetic sequence with diff={actual_diff}"
                else:
                    reward = 0.5
                    feedback = f"Valid arithmetic sequence but wrong diff (expected {expected_diff}, got {actual_diff})"
            else:
                reward = 1.5
                feedback = f"Valid arithmetic sequence with diff={actual_diff}"
            
            return VerificationResult(
                success=True,
                reward=reward,
                feedback=feedback,
                details={"difference": actual_diff, "sequence": sequence}
            )
        else:
            return VerificationResult(
                success=False,
                reward=-0.3,
                feedback="Not a valid arithmetic sequence",
                details={"differences": diffs, "sequence": sequence}
            )
    
    @staticmethod
    def verify_fibonacci_sequence(sequence: List[int]) -> VerificationResult:
        """Verify if sequence follows Fibonacci pattern"""
        if len(sequence) < 3:
            return VerificationResult(
                success=False,
                reward=-0.5,
                feedback="Sequence too short to verify",
                details={"length": len(sequence)}
            )
        
        # Check Fibonacci property
        errors = []
        for i in range(2, len(sequence)):
            if sequence[i] != sequence[i-1] + sequence[i-2]:
                errors.append(i)
        
        if not errors:
            return VerificationResult(
                success=True,
                reward=2.0,
                feedback="Perfect Fibonacci sequence",
                details={"sequence": sequence}
            )
        elif len(errors) <= len(sequence) * 0.1:  # Allow 10% errors
            return VerificationResult(
                success=True,
                reward=1.0,
                feedback=f"Mostly correct Fibonacci with {len(errors)} errors",
                details={"sequence": sequence, "error_positions": errors}
            )
        else:
            return VerificationResult(
                success=False,
                reward=-0.5,
                feedback="Not a valid Fibonacci sequence",
                details={"sequence": sequence, "error_positions": errors}
            )


class RLVRRewardComputer:
    """Main class for computing RLVR rewards"""
    
    def __init__(self):
        self.sql_verifier = SQLVerifier()
        self.code_verifier = CodeVerifier()
        self.pattern_verifier = PatternVerifier()
        
        # Reward shaping parameters
        self.length_penalty_factor = 0.01
        self.complexity_bonus_factor = 0.1
        self.kl_penalty_factor = 0.01
    
    def compute_reward(self, 
                      generated_text: str,
                      task_type: str,
                      reference_text: Optional[str] = None,
                      context: Optional[Dict] = None,
                      kl_divergence: Optional[float] = None) -> Tuple[float, VerificationResult]:
        """
        Compute RLVR reward for generated text
        
        Args:
            generated_text: The model's generated output
            task_type: Type of task (sql, code, pattern, etc.)
            reference_text: Optional reference/ground truth
            context: Additional context (schema, test cases, etc.)
            kl_divergence: KL divergence from reference model
        
        Returns:
            Tuple of (final_reward, verification_result)
        """
        
        # Get base reward from verification
        if task_type == "sql":
            verification = self._verify_sql(generated_text, context)
        elif task_type == "python":
            verification = self._verify_python(generated_text, context)
        elif task_type == "arithmetic":
            verification = self._verify_arithmetic(generated_text, context)
        elif task_type == "fibonacci":
            verification = self._verify_fibonacci(generated_text, context)
        else:
            # Default text generation reward
            verification = self._compute_text_reward(generated_text, reference_text)
        
        # Apply reward shaping
        final_reward = verification.reward
        
        # Length penalty (prefer concise solutions)
        length_penalty = -self.length_penalty_factor * len(generated_text.split())
        final_reward += length_penalty
        
        # KL penalty (prevent diverging too far from base model)
        if kl_divergence is not None:
            kl_penalty = -self.kl_penalty_factor * kl_divergence
            final_reward += kl_penalty
        
        # Clip reward to reasonable range
        final_reward = np.clip(final_reward, -3.0, 3.0)
        
        return final_reward, verification
    
    def _verify_sql(self, query: str, context: Dict) -> VerificationResult:
        """Verify SQL query with context"""
        if context and 'schema' in context:
            self.sql_verifier.setup_test_database(context['schema'])
        
        expected_result = context.get('expected_result') if context else None
        return self.sql_verifier.verify_sql_query(query, expected_result)
    
    def _verify_python(self, code: str, context: Dict) -> VerificationResult:
        """Verify Python code with test cases"""
        test_cases = context.get('test_cases', []) if context else []
        return self.code_verifier.verify_python_code(code, test_cases)
    
    def _verify_arithmetic(self, text: str, context: Dict) -> VerificationResult:
        """Verify arithmetic sequence"""
        # Parse numbers from text
        numbers = self._extract_numbers(text)
        expected_diff = context.get('difference') if context else None
        return self.pattern_verifier.verify_arithmetic_sequence(numbers, expected_diff)
    
    def _verify_fibonacci(self, text: str, context: Dict) -> VerificationResult:
        """Verify Fibonacci sequence"""
        numbers = self._extract_numbers(text)
        return self.pattern_verifier.verify_fibonacci_sequence(numbers)
    
    def _compute_text_reward(self, generated: str, reference: Optional[str]) -> VerificationResult:
        """Compute reward for general text generation"""
        if reference:
            # Simple similarity-based reward
            gen_tokens = set(generated.lower().split())
            ref_tokens = set(reference.lower().split())
            
            if gen_tokens == ref_tokens:
                reward = 2.0
            else:
                overlap = len(gen_tokens & ref_tokens)
                total = len(gen_tokens | ref_tokens)
                reward = overlap / total if total > 0 else 0
            
            return VerificationResult(
                success=reward > 0.5,
                reward=reward,
                feedback=f"Text similarity: {reward:.2f}",
                details={"generated": generated, "reference": reference}
            )
        else:
            # Just check if non-empty
            if generated.strip():
                return VerificationResult(
                    success=True,
                    reward=0.5,
                    feedback="Generated non-empty text",
                    details={"generated": generated}
                )
            else:
                return VerificationResult(
                    success=False,
                    reward=-1.0,
                    feedback="Generated empty text",
                    details={}
                )
    
    def _extract_numbers(self, text: str) -> List[int]:
        """Extract numbers from text"""
        numbers = re.findall(r'-?\d+', text)
        return [int(n) for n in numbers]
    
    def batch_compute_rewards(self, 
                             generated_texts: List[str],
                             task_types: List[str],
                             contexts: List[Optional[Dict]] = None,
                             kl_divergences: Optional[mx.array] = None) -> Tuple[mx.array, List[VerificationResult]]:
        """
        Compute rewards for a batch of generated texts
        
        Returns:
            Tuple of (reward_tensor, verification_results)
        """
        rewards = []
        verifications = []
        
        if contexts is None:
            contexts = [None] * len(generated_texts)
        
        kl_list = kl_divergences.tolist() if kl_divergences is not None else [None] * len(generated_texts)
        
        for i, (text, task_type, context, kl) in enumerate(zip(generated_texts, task_types, contexts, kl_list)):
            reward, verification = self.compute_reward(text, task_type, context=context, kl_divergence=kl)
            rewards.append(reward)
            verifications.append(verification)
        
        return mx.array(rewards), verifications


# Utility functions for reward normalization and shaping
def normalize_rewards(rewards: mx.array, eps: float = 1e-8) -> mx.array:
    """Normalize rewards to have zero mean and unit variance"""
    mean = mx.mean(rewards)
    std = mx.std(rewards)
    return (rewards - mean) / (std + eps)


def compute_advantages(rewards: mx.array, values: mx.array, gamma: float = 0.99, lam: float = 0.95) -> mx.array:
    """Compute GAE (Generalized Advantage Estimation) advantages"""
    advantages = mx.zeros_like(rewards)
    last_advantage = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value - values[t]
        advantages[t] = last_advantage = delta + gamma * lam * last_advantage
    
    return advantages


def compute_returns(rewards: mx.array, gamma: float = 0.99) -> mx.array:
    """Compute discounted returns"""
    returns = mx.zeros_like(rewards)
    running_return = 0
    
    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return
    
    return returns