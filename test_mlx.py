#!/usr/bin/env python
"""Test script to verify MLX installation"""

print("Testing MLX installation...")

# Test imports
try:
    import mlx.core as mx
    print("✓ MLX core imported successfully")
except ImportError as e:
    print(f"✗ Error importing MLX core: {e}")
    exit(1)

try:
    import mlx.nn as nn
    print("✓ MLX nn imported successfully")
except ImportError as e:
    print(f"✗ Error importing MLX nn: {e}")
    exit(1)

try:
    import mlx.optimizers as optim
    print("✓ MLX optimizers imported successfully")
except ImportError as e:
    print(f"✗ Error importing MLX optimizers: {e}")
    exit(1)

try:
    import matplotlib.pyplot as plt
    print("✓ Matplotlib imported successfully")
except ImportError as e:
    print(f"✗ Error importing matplotlib: {e}")
    exit(1)

# Test basic MLX operations
try:
    x = mx.array([1, 2, 3])
    y = mx.array([4, 5, 6])
    z = x + y
    print(f"✓ Basic MLX operations work: {x} + {y} = {z}")
except Exception as e:
    print(f"✗ Error with basic MLX operations: {e}")
    exit(1)

print("\nAll tests passed! MLX is properly installed.")
import sys
print(f"Python version: {sys.version}")
print(f"MLX device: {mx.default_device()}")