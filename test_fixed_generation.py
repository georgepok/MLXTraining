#!/usr/bin/env python3
"""Test the fixed SQL generation function"""

import sqlite3
from mlx_lm import load, generate

print("🧪 Testing Fixed SQL Generation")
print("=" * 70)

# Load model
print("\n📥 Loading model...")
model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
print("✅ Model loaded")

# Connect to database
conn = sqlite3.connect("rlvr_training.db")
print("✅ Database connected")

def generate_sql_fixed(question: str, model, tokenizer, max_tokens: int = 100) -> str:
    """Fixed SQL generation function"""
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

    # Extract SQL after "SQL:"
    if 'SQL:' in generated:
        generated = generated.split('SQL:', 1)[1].strip()

    # Extract just the SQL statement
    if ';' in generated:
        generated = generated.split(';')[0] + ';'
    else:
        if '\n' in generated:
            generated = generated.split('\n')[0].strip()
        if not generated.endswith(';'):
            generated = generated + ';'

    return generated.strip()

# Test
test_questions = [
    "List all employees",
    "Count total employees",
    "Find employees in Engineering"
]

print("\n" + "=" * 70)
print("TESTING FIXED EXTRACTION")
print("=" * 70)

success_count = 0
for i, q in enumerate(test_questions, 1):
    print(f"\n{i}. Question: {q}")

    sql = generate_sql_fixed(q, model, tokenizer, max_tokens=50)
    print(f"   Extracted SQL: {sql}")

    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        print(f"   ✅ Success! {len(results)} rows returned")
        success_count += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "=" * 70)
print(f"Results: {success_count}/{len(test_questions)} successful ({success_count/len(test_questions)*100:.0f}%)")
print("=" * 70)

conn.close()
