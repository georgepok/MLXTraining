#!/usr/bin/env python3
"""
Comprehensive evaluation of training results
Compares base model vs potentially trained model
"""

import sqlite3
import numpy as np
from mlx_lm import load, generate
from dataclasses import dataclass
from typing import List, Tuple

print("="*70)
print("COMPREHENSIVE TRAINING EVALUATION")
print("="*70)

# Setup database
conn = sqlite3.connect("rlvr_training.db")
cursor = conn.cursor()

@dataclass
class SQLExample:
    question: str
    correct_sql: str
    category: str

# Comprehensive test set
test_examples = [
    # Basic SELECT
    SQLExample("List all employees", "SELECT * FROM employees;", "basic"),
    SQLExample("Show all departments", "SELECT * FROM departments;", "basic"),
    SQLExample("List all projects", "SELECT * FROM projects;", "basic"),

    # WHERE clauses
    SQLExample("Find employees in Engineering", "SELECT * FROM employees WHERE department = 'Engineering';", "where"),
    SQLExample("Find employees in Marketing", "SELECT * FROM employees WHERE department = 'Marketing';", "where"),
    SQLExample("Find active employees", "SELECT * FROM employees WHERE is_active = 1;", "where"),

    # Aggregates
    SQLExample("Count total employees", "SELECT COUNT(*) FROM employees;", "aggregate"),
    SQLExample("Count Engineering employees", "SELECT COUNT(*) FROM employees WHERE department = 'Engineering';", "aggregate"),
    SQLExample("Calculate average salary", "SELECT AVG(salary) FROM employees;", "aggregate"),
    SQLExample("Find highest salary", "SELECT MAX(salary) FROM employees;", "aggregate"),

    # Complex
    SQLExample("Find employees earning over 80000", "SELECT * FROM employees WHERE salary > 80000;", "complex"),
    SQLExample("Count active projects", "SELECT COUNT(*) FROM projects WHERE status = 'active';", "complex"),
    SQLExample("List employees by salary descending", "SELECT * FROM employees ORDER BY salary DESC;", "complex"),
]

def generate_sql_fixed(question: str, model, tokenizer, max_tokens: int = 100) -> str:
    """Fixed SQL generation"""
    prompt = f"""Generate a SQL query for: {question}

SQL:"""

    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False
    )

    generated = output.strip()

    if 'SQL:' in generated:
        generated = generated.split('SQL:', 1)[1].strip()

    if ';' in generated:
        generated = generated.split(';')[0] + ';'
    else:
        if '\n' in generated:
            generated = generated.split('\n')[0].strip()
        if not generated.endswith(';'):
            generated = generated + ';'

    return generated.strip()

def evaluate_sql(sql: str, correct_sql: str) -> Tuple[bool, float, str]:
    """
    Evaluate generated SQL
    Returns: (success, reward, status)
    """
    try:
        cursor.execute(sql)
        results = cursor.fetchall()

        # Base reward for execution
        reward = 0.5

        # Bonus for results
        if results:
            reward += 0.3
            if 1 <= len(results) <= 20:
                reward += 0.2
        else:
            reward += 0.1

        # Correctness bonus
        if sql.strip().lower() == correct_sql.strip().lower():
            reward += 0.5
        elif correct_sql.strip().lower() in sql.strip().lower():
            reward += 0.3

        return True, reward, f"✓ {len(results)} rows"
    except sqlite3.Error as e:
        error_msg = str(e).lower()
        if 'syntax' in error_msg:
            reward = -0.5
        elif 'no such table' in error_msg or 'no such column' in error_msg:
            reward = -0.3
        else:
            reward = -0.2
        return False, reward, f"✗ {str(e)[:40]}"

def evaluate_model(model, tokenizer, examples: List[SQLExample], model_name: str):
    """Evaluate model on test set"""
    print(f"\n{'='*70}")
    print(f"EVALUATING: {model_name}")
    print(f"{'='*70}")

    results = []
    category_results = {}

    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example.question}")
        print(f"   Category: {example.category}")

        # Generate SQL
        generated_sql = generate_sql_fixed(example.question, model, tokenizer, max_tokens=100)
        print(f"   Generated: {generated_sql}")
        print(f"   Expected:  {example.correct_sql}")

        # Evaluate
        success, reward, status = evaluate_sql(generated_sql, example.correct_sql)
        print(f"   Status: {status}")
        print(f"   Reward: {reward:+.2f}")

        # Record
        result = {
            'question': example.question,
            'category': example.category,
            'generated': generated_sql,
            'expected': example.correct_sql,
            'success': success,
            'reward': reward,
            'exact_match': generated_sql.strip().lower() == example.correct_sql.strip().lower()
        }
        results.append(result)

        # Track by category
        if example.category not in category_results:
            category_results[example.category] = []
        category_results[example.category].append(result)

    # Summary statistics
    print(f"\n{'='*70}")
    print(f"SUMMARY: {model_name}")
    print(f"{'='*70}")

    total = len(results)
    successful = sum(1 for r in results if r['success'])
    exact_matches = sum(1 for r in results if r['exact_match'])
    avg_reward = np.mean([r['reward'] for r in results])

    print(f"\nOverall Performance:")
    print(f"  Total examples: {total}")
    print(f"  Successful executions: {successful}/{total} ({successful/total*100:.1f}%)")
    print(f"  Exact matches: {exact_matches}/{total} ({exact_matches/total*100:.1f}%)")
    print(f"  Average reward: {avg_reward:+.3f}")

    print(f"\nPerformance by Category:")
    for category, cat_results in sorted(category_results.items()):
        cat_success = sum(1 for r in cat_results if r['success'])
        cat_exact = sum(1 for r in cat_results if r['exact_match'])
        cat_reward = np.mean([r['reward'] for r in cat_results])
        print(f"  {category:12s}: {cat_success}/{len(cat_results)} success, "
              f"{cat_exact}/{len(cat_results)} exact, "
              f"reward={cat_reward:+.3f}")

    return results, {
        'success_rate': successful/total,
        'exact_match_rate': exact_matches/total,
        'avg_reward': avg_reward,
        'category_results': category_results
    }

# Load base model (fresh, no training)
print("\n📥 Loading base model (no training)...")
base_model, base_tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
print("✅ Base model loaded")

# Evaluate base model
base_results, base_stats = evaluate_model(base_model, base_tokenizer, test_examples, "BASE MODEL (Untrained)")

# Check if we have a trained model
print("\n" + "="*70)
print("Checking for trained model with LoRA adapters...")
print("="*70)

# Try to load the model that should have been trained in the notebook
# Since we can't directly access notebook state, we'll note this
print("\n⚠️  Note: To compare with trained model, the notebook training must complete first")
print("    and LoRA parameters must be applied to the model.")
print("\n    For now, showing baseline (untrained) performance.")

print("\n" + "="*70)
print("ANALYSIS")
print("="*70)

print(f"\nBase Model Performance:")
print(f"  Success Rate: {base_stats['success_rate']*100:.1f}%")
print(f"  Exact Match Rate: {base_stats['exact_match_rate']*100:.1f}%")
print(f"  Average Reward: {base_stats['avg_reward']:+.3f}")

print(f"\n🎯 Expected after training:")
print(f"  Success Rate: 70-85% (target)")
print(f"  Exact Match Rate: 30-50% (target)")
print(f"  Average Reward: +0.6 to +0.9 (target)")

print(f"\n📊 Current Gap to Target:")
gap_success = 75 - (base_stats['success_rate']*100)
gap_reward = 0.75 - base_stats['avg_reward']
print(f"  Success Rate Gap: {gap_success:+.1f}%")
print(f"  Reward Gap: {gap_reward:+.3f}")

if base_stats['success_rate'] < 0.5:
    print(f"\n⚠️  LOW BASELINE PERFORMANCE DETECTED")
    print(f"  The base model is struggling with SQL generation.")
    print(f"  Training should significantly improve this.")
elif base_stats['success_rate'] > 0.8:
    print(f"\n✅ HIGH BASELINE PERFORMANCE")
    print(f"  The base model is already quite good at SQL.")
    print(f"  Training improvements will be more subtle.")
else:
    print(f"\n✓ MODERATE BASELINE PERFORMANCE")
    print(f"  Good baseline for measuring training impact.")

conn.close()
print("\n" + "="*70)
print("EVALUATION COMPLETE")
print("="*70)
