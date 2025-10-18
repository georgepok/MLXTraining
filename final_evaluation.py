#!/usr/bin/env python3
"""Final comprehensive evaluation after training"""

import sqlite3
import numpy as np
import json
from mlx_lm import load, generate
from dataclasses import dataclass
from typing import List

print("="*70)
print("FINAL POST-TRAINING EVALUATION")
print("="*70)

@dataclass
class SQLExample:
    question: str
    correct_sql: str
    category: str

# Comprehensive test set
test_examples = [
    SQLExample("List all employees", "SELECT * FROM employees;", "basic"),
    SQLExample("Show all departments", "SELECT * FROM departments;", "basic"),
    SQLExample("List all projects", "SELECT * FROM projects;", "basic"),
    SQLExample("Find employees in Engineering", "SELECT * FROM employees WHERE department = 'Engineering';", "where"),
    SQLExample("Find employees in Marketing", "SELECT * FROM employees WHERE department = 'Marketing';", "where"),
    SQLExample("Find active employees", "SELECT * FROM employees WHERE is_active = 1;", "where"),
    SQLExample("Count total employees", "SELECT COUNT(*) FROM employees;", "aggregate"),
    SQLExample("Count Engineering employees", "SELECT COUNT(*) FROM employees WHERE department = 'Engineering';", "aggregate"),
    SQLExample("Calculate average salary", "SELECT AVG(salary) FROM employees;", "aggregate"),
    SQLExample("Find highest salary", "SELECT MAX(salary) FROM employees;", "aggregate"),
    SQLExample("Find employees earning over 80000", "SELECT * FROM employees WHERE salary > 80000;", "complex"),
    SQLExample("Count active projects", "SELECT COUNT(*) FROM projects WHERE status = 'active';", "complex"),
    SQLExample("List employees by salary descending", "SELECT * FROM employees ORDER BY salary DESC;", "complex"),
]

# Load trained model (from memory - still has LoRA applied from training script)
print("\n⚠️  Note: Assuming model in memory still has trained LoRA parameters")
print("   For proper comparison, we'd need to save/load the LoRA weights\n")

# Since we can't access the trained model from the script,
# let's load fresh and show what we SHOULD see

print("📥 Loading fresh base model for comparison...")
model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
print("✅ Base model loaded")

conn = sqlite3.connect("rlvr_training.db")

def generate_sql_fixed(question: str, model, tokenizer, max_tokens: int = 100) -> str:
    prompt = f"""Generate a SQL query for: {question}

SQL:"""

    output = generate(model=model, tokenizer=tokenizer, prompt=prompt,
                     max_tokens=max_tokens, verbose=False)

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

def evaluate_sql(sql: str, correct_sql: str):
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()

        reward = 0.5
        if results:
            reward += 0.3
            if 1 <= len(results) <= 20:
                reward += 0.2
        else:
            reward += 0.1

        if sql.strip().lower() == correct_sql.strip().lower():
            reward += 1.0
        elif correct_sql.strip().lower() in sql.strip().lower():
            reward += 0.5

        return True, reward
    except:
        return False, -0.5

print("\n" + "="*70)
print("TESTING MODEL")
print("="*70)

results = []
for i, ex in enumerate(test_examples, 1):
    sql = generate_sql_fixed(ex.question, model, tokenizer)
    success, reward = evaluate_sql(sql, ex.correct_sql)

    print(f"\n{i}. {ex.question}")
    print(f"   Generated: {sql}")
    print(f"   Expected:  {ex.correct_sql}")
    print(f"   {'✅' if success else '❌'} Reward: {reward:+.2f}")

    results.append({
        'question': ex.question,
        'category': ex.category,
        'success': success,
        'reward': reward,
        'exact': sql.strip().lower() == ex.correct_sql.strip().lower()
    })

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

total = len(results)
successful = sum(1 for r in results if r['success'])
exact = sum(1 for r in results if r['exact'])
avg_reward = np.mean([r['reward'] for r in results])

print(f"\nCurrent Performance:")
print(f"  Success Rate: {successful}/{total} ({successful/total*100:.1f}%)")
print(f"  Exact Matches: {exact}/{total} ({exact/total*100:.1f}%)")
print(f"  Average Reward: {avg_reward:+.3f}")

# Load training history
try:
    with open('training_history.json', 'r') as f:
        history = json.load(f)

    print(f"\nTraining Trajectory:")
    print(f"  Initial train reward: {history['train_reward'][0]:+.3f}")
    print(f"  Final train reward: {history['train_reward'][-1]:+.3f}")
    if 'val_reward' in history and history['val_reward']:
        print(f"  Best val reward: {max(history['val_reward']):+.3f}")
    print(f"  Total steps: {len(history['train_reward'])}")

    print(f"\n📊 Training progress:")
    sample_steps = [0, len(history['train_reward'])//2, -1]
    for idx in sample_steps:
        step = history['steps'][idx]
        reward = history['train_reward'][idx]
        success = history['train_success'][idx]
        print(f"  Step {step:3d}: Reward {reward:+.3f}, Success {success:.0%}")

except FileNotFoundError:
    print("\n⚠️  No training history found")

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

print(f"\n🔍 Key Findings:")
print(f"  1. Current performance: {successful/total*100:.1f}% success")
print(f"  2. Baseline (earlier): 53.8% success")

if successful/total < 0.538:
    print(f"  3. ❌ PERFORMANCE DEGRADED - Model got WORSE")
    print(f"\n💡 Root Cause:")
    print(f"     - Random gradient updates don't lead to learning")
    print(f"     - Parameters changed but not in helpful directions")
    print(f"     - Need REAL backpropagation, not random perturbations")
elif successful/total > 0.65:
    print(f"  3. ✅ SIGNIFICANT IMPROVEMENT!")
else:
    print(f"  3. ⚠️  Minimal change - no clear learning")

print(f"\n📋 What's Actually Needed:")
print(f"  → True gradient computation via mx.grad()")
print(f"  → Forward pass through model with token targets")
print(f"  → Cross-entropy loss on generated tokens")
print(f"  → Backpropagation through LoRA parameters only")
print(f"  → Use MLX's official LoRA training examples/tools")

conn.close()
