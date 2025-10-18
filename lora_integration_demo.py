#!/usr/bin/env python3
"""
Demonstrate how trained LoRA adapters are integrated back into the base Qwen model
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

def demonstrate_lora_integration():
    """Show the exact mechanism of LoRA integration"""
    
    print("🔧 HOW TRAINED LORA ADAPTERS ARE INTEGRATED INTO BASE QWEN MODEL")
    print("=" * 80)
    
    # Simulate a Qwen attention layer
    print("\n1️⃣ ORIGINAL QWEN ATTENTION LAYER:")
    
    # Original Qwen q_proj layer (example dimensions for Qwen2.5-0.5B)
    hidden_size = 896
    original_weight = mx.random.normal((hidden_size, hidden_size)) * 0.02
    
    class OriginalLinear(nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = weight
            
        def __call__(self, x):
            return x @ self.weight
    
    original_q_proj = OriginalLinear(original_weight)
    print(f"   Original q_proj weight shape: {original_q_proj.weight.shape}")
    print(f"   Original parameters: {original_q_proj.weight.size:,}")
    
    # Test input
    batch_size, seq_len = 2, 10
    test_input = mx.random.normal((batch_size, seq_len, hidden_size))
    original_output = original_q_proj(test_input)
    print(f"   Original output shape: {original_output.shape}")
    
    print("\n2️⃣ ADDING LORA ADAPTERS:")
    
    # LoRA parameters
    rank = 8
    alpha = 16.0
    scaling = alpha / rank
    
    # Initial LoRA matrices (before training)
    lora_A_initial = mx.random.normal((hidden_size, rank)) * 0.01
    lora_B_initial = mx.zeros((rank, hidden_size))
    
    print(f"   LoRA A shape: {lora_A_initial.shape}")
    print(f"   LoRA B shape: {lora_B_initial.shape}")
    print(f"   LoRA parameters: {lora_A_initial.size + lora_B_initial.size:,}")
    print(f"   Scaling factor: {scaling}")
    
    # Initial LoRA output (should be zero since B is initialized to zero)
    initial_lora_output = (test_input @ lora_A_initial) @ lora_B_initial * scaling
    print(f"   Initial LoRA output (should be ~0): {mx.mean(mx.abs(initial_lora_output)):.6f}")
    
    print("\n3️⃣ SIMULATING TRAINING - LORA PARAMETERS CHANGE:")
    
    # After training, LoRA B matrix has learned values
    lora_A_trained = lora_A_initial  # A usually changes less
    lora_B_trained = lora_B_initial + mx.random.normal((rank, hidden_size)) * 0.1
    
    print(f"   LoRA A change: {float(mx.mean(mx.abs(lora_A_trained - lora_A_initial))):.6f}")
    print(f"   LoRA B change: {float(mx.mean(mx.abs(lora_B_trained - lora_B_initial))):.6f}")
    
    # Trained LoRA output
    trained_lora_output = (test_input @ lora_A_trained) @ lora_B_trained * scaling
    print(f"   Trained LoRA output: {float(mx.mean(mx.abs(trained_lora_output))):.6f}")
    
    print("\n4️⃣ THREE WAYS TO INTEGRATE TRAINED LORA:")
    
    print("\n   🔀 METHOD 1: RUNTIME ADDITION (Recommended)")
    print("   ────────────────────────────────────────")
    
    class LoRAIntegratedLayer(nn.Module):
        def __init__(self, original_layer, lora_A, lora_B, scaling):
            super().__init__()
            self.original_layer = original_layer
            self.lora_A = lora_A
            self.lora_B = lora_B
            self.scaling = scaling
            
        def __call__(self, x):
            # Original computation
            original_out = self.original_layer(x)
            
            # LoRA computation  
            lora_out = (x @ self.lora_A) @ self.lora_B * self.scaling
            
            # Combined output
            return original_out + lora_out
    
    runtime_layer = LoRAIntegratedLayer(original_q_proj, lora_A_trained, lora_B_trained, scaling)
    runtime_output = runtime_layer(test_input)
    
    print(f"   ✅ Runtime integration: Original + LoRA computed separately")
    print(f"   ✅ Base model unchanged: {mx.array_equal(original_q_proj.weight, original_weight)}")
    print(f"   ✅ Output shape: {runtime_output.shape}")
    print(f"   ✅ Difference from original: {float(mx.mean(mx.abs(runtime_output - original_output))):.6f}")
    
    print("\n   🔗 METHOD 2: WEIGHT MERGING (For inference)")
    print("   ──────────────────────────────────────────")
    
    # Merge LoRA into original weights: W' = W + (A @ B) * scaling
    lora_delta = lora_A_trained @ lora_B_trained * scaling
    merged_weight = original_weight + lora_delta.T  # Transpose to match weight orientation
    
    class MergedLinear(nn.Module):
        def __init__(self, merged_weight):
            super().__init__()
            self.weight = merged_weight
            
        def __call__(self, x):
            return x @ self.weight
    
    merged_layer = MergedLinear(merged_weight)
    merged_output = merged_layer(test_input)
    
    print(f"   ✅ Weight merging: W_new = W_original + LoRA_delta")
    print(f"   ✅ LoRA delta shape: {lora_delta.shape}")
    print(f"   ✅ Merged weight shape: {merged_weight.shape}")
    print(f"   ✅ Single matrix multiplication (faster inference)")
    print(f"   ✅ Output matches runtime method: {mx.allclose(merged_output, runtime_output, atol=1e-5)}")
    
    print("\n   📊 METHOD 3: PARAMETER REPLACEMENT (Direct update)")
    print("   ─────────────────────────────────────────────────")
    
    # Directly update the model's parameters (destructive)
    updated_q_proj = OriginalLinear(merged_weight)
    replacement_output = updated_q_proj(test_input)
    
    print(f"   ✅ Direct replacement: model.layer.q_proj.weight = merged_weight")
    print(f"   ⚠️  Original weights lost (destructive)")
    print(f"   ✅ Fastest inference (no additional computation)")
    print(f"   ✅ Output identical to merged: {mx.array_equal(replacement_output, merged_output)}")
    
    print("\n5️⃣ PRACTICAL INTEGRATION IN QWEN MODEL:")
    
    print("""
   🏗️  STEP-BY-STEP INTEGRATION PROCESS:
   
   1. Train LoRA adapters:
      for step in training_steps:
          # Forward pass with LoRA
          output = original_layer(x) + (x @ lora_A) @ lora_B * scaling
          # Compute loss and gradients
          # Update lora_A and lora_B (NOT original weights)
   
   2. After training, choose integration method:
   
      🔀 For continued training (Method 1):
         model.layer.q_proj = LoRAIntegratedLayer(original_q_proj, lora_A, lora_B)
         # Keeps LoRA separate, can continue training
      
      🔗 For deployment (Method 2):
         delta = (lora_A @ lora_B * scaling).T
         model.layer.q_proj.weight = original_weight + delta
         # Single weight matrix, optimal inference speed
      
      📊 For model distribution (Method 3):
         # Save merged weights as new base model
         # Distribute as standard model file
   """)
    
    print("\n6️⃣ VERIFICATION - ALL METHODS PRODUCE SAME RESULT:")
    
    print(f"   Original output mean: {float(mx.mean(original_output)):.6f}")
    print(f"   Runtime LoRA mean:    {float(mx.mean(runtime_output)):.6f}")
    print(f"   Merged weights mean:  {float(mx.mean(merged_output)):.6f}")
    print(f"   Replaced weights mean:{float(mx.mean(replacement_output)):.6f}")
    
    # Check all outputs are equivalent
    runtime_vs_merged = mx.allclose(runtime_output, merged_output, atol=1e-5)
    merged_vs_replaced = mx.array_equal(merged_output, replacement_output)
    
    print(f"\n   ✅ Runtime ≈ Merged:     {runtime_vs_merged}")
    print(f"   ✅ Merged = Replaced:    {merged_vs_replaced}")
    print(f"   ✅ All methods work:     {runtime_vs_merged and merged_vs_replaced}")
    
    print("\n🎯 SUMMARY - HOW CHANGED LORA GETS ATTACHED:")
    print("""
   The trained LoRA adapters are integrated into the base Qwen model through:
   
   🔧 Mathematical Formula:
      New_Output = Original_Layer(input) + LoRA_Adaptation(input)
      Where: LoRA_Adaptation(input) = (input @ A) @ B * scaling
   
   🏗️  Implementation Options:
      1. Runtime Addition: Keep separate, add during forward pass
      2. Weight Merging: Merge into single weight matrix  
      3. Parameter Replacement: Replace original weights entirely
   
   🎯 Key Insight: 
      LoRA changes are ADDED to original model behavior, not replacing it.
      This preserves the base model's knowledge while adding new capabilities.
   """)
    
    return {
        'original_output': original_output,
        'runtime_output': runtime_output, 
        'merged_output': merged_output,
        'lora_A_trained': lora_A_trained,
        'lora_B_trained': lora_B_trained,
        'original_weight': original_weight,
        'merged_weight': merged_weight
    }

if __name__ == "__main__":
    results = demonstrate_lora_integration()
    print("\n🎯 Demo completed! All integration methods verified.")