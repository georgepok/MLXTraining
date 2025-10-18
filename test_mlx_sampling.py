#!/usr/bin/env python3
"""Test if mlx_lm.generate supports sampling parameters"""

import sys
from inspect import signature, getfullargspec

try:
    from mlx_lm import generate
    import mlx_lm
    
    print("="*70)
    print("MLX-LM Sampling Capability Test")
    print("="*70)
    
    # Check version
    if hasattr(mlx_lm, '__version__'):
        print(f"\nmlx_lm version: {mlx_lm.__version__}")
    else:
        print("\nmlx_lm version: Unknown (no __version__ attribute)")
    
    # Check generate function signature
    print(f"\n{'─'*70}")
    print("generate() function signature:")
    print(f"{'─'*70}")
    
    try:
        sig = signature(generate)
        print(f"{sig}")
        
        # Get parameter names
        params = list(sig.parameters.keys())
        print(f"\nParameters: {', '.join(params)}")
        
        # Check for sampling-related parameters
        sampling_params = ['temp', 'temperature', 'top_p', 'top_k', 'sampler', 'sampling']
        found_sampling = [p for p in sampling_params if p in params]
        
        if found_sampling:
            print(f"\n✅ Sampling parameters found: {', '.join(found_sampling)}")
        else:
            print(f"\n❌ No standard sampling parameters found")
            print(f"   Checked for: {', '.join(sampling_params)}")
        
    except Exception as e:
        print(f"Error inspecting signature: {e}")
        
        # Try alternative method
        try:
            spec = getfullargspec(generate)
            print(f"\nArgs: {spec.args}")
            print(f"Defaults: {spec.defaults}")
        except Exception as e2:
            print(f"Alternative method also failed: {e2}")
    
    print(f"\n{'='*70}")
    print("DIAGNOSIS:")
    print(f"{'='*70}")
    
    if not found_sampling:
        print("\n⚠️  Current mlx_lm version does NOT support sampling!")
        print("\nThis explains why GRPO shows 0% improvement:")
        print("  • generate() uses greedy decoding (deterministic)")
        print("  • All completions in a group are identical")
        print("  • Advantages = 0 → gradients = 0 → no learning")
        
        print("\n💡 Solutions:")
        print("  1. Upgrade mlx_lm to version with sampling support")
        print("  2. Implement custom generate with temperature")
        print("  3. Use prompt variations (quick workaround)")
    else:
        print("\n✅ Sampling IS supported!")
        print(f"\nAvailable parameters: {', '.join(found_sampling)}")
        print("\n💡 Fix: Add temperature to generate() calls in GRPO")
        print("   Example: generate(..., temp=0.7)")
    
    print(f"\n{'='*70}\n")
    
except ImportError as e:
    print(f"Error: Could not import mlx_lm")
    print(f"Details: {e}")
    sys.exit(1)
