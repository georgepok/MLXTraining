#!/usr/bin/env python3
"""
Diagnostic script to check what the model is actually generating
"""

import mlx.core as mx
import sqlite3
from mlx_lm import load, generate

print("🔍 TRAINING DIAGNOSTIC")
print("=" * 70)

# Load model and tokenizer
print("\n📥 Loading model...")
model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
print("✅ Model loaded")

# Connect to database
print("\n🗃️ Connecting to database...")
conn = sqlite3.connect("rlvr_training.db")
print("✅ Database connected")

# Test questions
test_questions = [
    "List all employees",
    "Count total employees",
    "Find employees in Engineering"
]

print("\n" + "=" * 70)
print("GENERATION TESTS")
print("=" * 70)

for i, q in enumerate(test_questions, 1):
    print(f"\n{'─' * 70}")
    print(f"Test {i}: {q}")
    print(f"{'─' * 70}")

    # Create prompt
    prompt = f"""Generate a SQL query for: {q}

SQL:"""

    print(f"\n📝 Prompt ({len(prompt)} chars):")
    print(f"   '{prompt}'")

    # Generate with model
    print(f"\n⚙️ Generating with max_tokens=100...")
    raw_output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=100,
        verbose=False
    )

    print(f"\n📤 Raw output ({len(raw_output)} chars):")
    print(f"   '{raw_output}'")

    # Extract SQL
    generated = raw_output[len(prompt):].strip()

    # Apply extraction logic
    if '\n' in generated:
        generated = generated.split('\n')[0]
    if ';' in generated:
        generated = generated.split(';')[0] + ';'
    else:
        generated = generated + ';'

    generated = generated.strip()

    print(f"\n🔧 Extracted SQL:")
    print(f"   '{generated}'")

    # Try to execute
    print(f"\n🔍 Execution test:")
    try:
        cursor = conn.cursor()
        cursor.execute(generated)
        results = cursor.fetchall()
        print(f"   ✅ Success! Returned {len(results)} rows")
        if results and len(results) <= 3:
            print(f"   Sample results: {results}")
    except sqlite3.Error as e:
        print(f"   ❌ SQL Error: {e}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)

# Close connection
conn.close()
