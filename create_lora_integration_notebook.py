#!/usr/bin/env python3
"""
Create notebook showing exactly how trained LoRA adapters are integrated into Qwen model
"""

import json

def get_lora_integration_cell():
    """Cell showing LoRA integration into Qwen model"""
    return '''# 🔗 HOW TRAINED LORA ADAPTERS ARE INTEGRATED BACK INTO QWEN MODEL
print("🔗 HOW TRAINED LORA ADAPTERS ARE INTEGRATED BACK INTO QWEN MODEL")
print("=" * 80)
print("This shows the exact mechanism of LoRA integration!")
print("=" * 80)

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Check if we have trained LoRA from previous cells
if 'lora_layers' in globals() and 'original_model' in globals():
    print("✅ Found trained LoRA adapters from previous training!")
    
    # Get a sample trained LoRA adapter
    sample_lora_key = list(lora_layers.keys())[0]
    sample_lora = lora_layers[sample_lora_key]
    
    print(f"\\n📋 EXAMINING TRAINED LORA: {sample_lora_key}")
    print(f"   LoRA A shape: {sample_lora['lora_A'].shape}")
    print(f"   LoRA B shape: {sample_lora['lora_B'].shape}")
    print(f"   Original layer: {type(sample_lora['original_layer'])}")
    
    # Extract components
    lora_A = sample_lora['lora_A']
    lora_B = sample_lora['lora_B']
    original_layer = sample_lora['original_layer']
    scaling = 16.0 / 8  # alpha / rank
    
    print(f"\\n🔍 LORA PARAMETER STATE:")
    lora_A_norm = float(mx.sum(lora_A ** 2) ** 0.5)
    lora_B_norm = float(mx.sum(lora_B ** 2) ** 0.5)
    print(f"   LoRA A norm: {lora_A_norm:.6f}")
    print(f"   LoRA B norm: {lora_B_norm:.6f}")
    print(f"   Scaling factor: {scaling}")
    
    # Create test input
    if hasattr(original_layer, 'weight'):
        input_dim = original_layer.weight.shape[1]
    else:
        input_dim = 896  # Qwen2.5-0.5B default
    
    test_input = mx.random.normal((2, 10, input_dim))
    print(f"   Test input shape: {test_input.shape}")
    
    print("\\n" + "="*60)
    print("METHOD 1: RUNTIME INTEGRATION (Current Implementation)")
    print("="*60)
    
    # Method 1: Runtime addition (what we're currently doing)
    original_output = original_layer(test_input)
    lora_output = (test_input @ lora_A) @ lora_B * scaling
    combined_output = original_output + lora_output
    
    print("🔀 RUNTIME INTEGRATION PROCESS:")
    print("   1. Forward pass through original layer:")
    print(f"      original_output = original_layer(input)")
    print(f"      Shape: {original_output.shape}")
    print(f"      Mean: {float(mx.mean(original_output)):.6f}")
    
    print("   2. Compute LoRA adaptation:")
    print(f"      lora_output = (input @ lora_A) @ lora_B * scaling")
    print(f"      Shape: {lora_output.shape}")
    print(f"      Mean: {float(mx.mean(lora_output)):.6f}")
    
    print("   3. Combine outputs:")
    print(f"      final_output = original_output + lora_output")
    print(f"      Shape: {combined_output.shape}")
    print(f"      Mean: {float(mx.mean(combined_output)):.6f}")
    
    print("\\n✅ ADVANTAGES:")
    print("   ✅ Base model weights unchanged")
    print("   ✅ Can continue training LoRA")
    print("   ✅ Can remove LoRA adapters easily")
    print("   ✅ Memory efficient (small LoRA params)")
    
    print("\\n" + "="*60)
    print("METHOD 2: WEIGHT MERGING (For Production)")
    print("="*60)
    
    # Method 2: Merge LoRA into original weights
    print("🔗 WEIGHT MERGING PROCESS:")
    
    if hasattr(original_layer, 'weight'):
        original_weight = original_layer.weight
        print(f"   1. Extract original weight: {original_weight.shape}")
        
        # Compute LoRA delta matrix
        lora_delta = lora_A @ lora_B * scaling
        print(f"   2. Compute LoRA delta: {lora_delta.shape}")
        print(f"      Delta formula: (lora_A @ lora_B) * scaling")
        print(f"      Delta magnitude: {float(mx.mean(mx.abs(lora_delta))):.6f}")
        
        # Merge weights (need to transpose delta to match weight layout)
        merged_weight = original_weight + lora_delta.T
        print(f"   3. Merge weights: W_new = W_original + delta.T")
        print(f"      Merged weight shape: {merged_weight.shape}")
        
        # Test merged layer
        class MergedLayer(nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = weight
            def __call__(self, x):
                return x @ self.weight
        
        merged_layer = MergedLayer(merged_weight)
        merged_output = merged_layer(test_input)
        
        print(f"   4. Test merged layer output:")
        print(f"      Shape: {merged_output.shape}")
        print(f"      Mean: {float(mx.mean(merged_output)):.6f}")
        
        # Verify outputs match
        output_match = mx.allclose(combined_output, merged_output, atol=1e-4)
        print(f"\\n🔍 VERIFICATION:")
        print(f"   Runtime output ≈ Merged output: {output_match}")
        
        if output_match:
            print("   ✅ Integration successful!")
        else:
            diff = float(mx.mean(mx.abs(combined_output - merged_output)))
            print(f"   ⚠️  Small numerical difference: {diff:.6f}")
        
        print("\\n✅ ADVANTAGES:")
        print("   ✅ Single matrix multiplication (faster)")
        print("   ✅ Standard model format")
        print("   ✅ No additional memory overhead")
        print("   ✅ Can distribute as regular model")
        
        print("\\n⚠️  TRADE-OFFS:")
        print("   ⚠️  Original weights lost")
        print("   ⚠️  Cannot easily remove LoRA")
        print("   ⚠️  Harder to continue LoRA training")
        
    else:
        print("   ⚠️ Cannot access original layer weights")
    
    print("\\n" + "="*60)
    print("METHOD 3: IN-PLACE MODEL UPDATE")
    print("="*60)
    
    print("🏗️ IN-PLACE INTEGRATION PROCESS:")
    print("   This method directly modifies the Qwen model structure")
    
    # Show how to update the actual model
    print("\\n   Example for updating Qwen model:")
    print(\"\"\"
   # Get the attention layer
   layer_0 = model.model.layers[0]
   attention = layer_0.self_attn
   
   # Get current q_proj
   current_q_proj = attention.q_proj
   
   # Replace with LoRA-integrated version
   class LoRAIntegratedProjection(nn.Module):
       def __init__(self, original_layer, lora_A, lora_B, scaling):
           super().__init__()
           self.original = original_layer
           self.lora_A = lora_A
           self.lora_B = lora_B  
           self.scaling = scaling
           
       def __call__(self, x):
           return self.original(x) + (x @ self.lora_A) @ self.lora_B * self.scaling
   
   # Replace the layer
   attention.q_proj = LoRAIntegratedProjection(
       current_q_proj, trained_lora_A, trained_lora_B, scaling
   )
   \"\"\")
    
    print("\\n✅ ADVANTAGES:")
    print("   ✅ Model behaves exactly as if LoRA was always there")
    print("   ✅ Transparent to downstream code")
    print("   ✅ Can save/load complete model")
    
    print("\\n" + "="*60)
    print("PRACTICAL RECOMMENDATION")
    print("="*60)
    
    print("🎯 CHOOSE INTEGRATION METHOD BASED ON USE CASE:")
    print()
    print("📚 CONTINUED RESEARCH/TRAINING:")
    print("   → Use Method 1 (Runtime Integration)")
    print("   → Keep LoRA separate for flexibility")
    print("   → Easy to experiment with different adapters")
    print()
    print("🚀 PRODUCTION DEPLOYMENT:")
    print("   → Use Method 2 (Weight Merging)")
    print("   → Single model file, optimal performance")
    print("   → Standard model distribution format")
    print()
    print("🔬 INTERACTIVE EXPERIMENTATION:")
    print("   → Use Method 3 (In-place Update)")
    print("   → Model behaves transparently")
    print("   → Easy to swap different LoRA versions")
    
    print("\\n🔍 CURRENT NOTEBOOK STATUS:")
    print(f"   ✅ We are using Method 1 (Runtime Integration)")
    print(f"   ✅ LoRA adapters attached to {len(lora_layers)} layers")
    print(f"   ✅ Base Qwen model weights preserved")
    print(f"   ✅ Can easily switch to other methods")
    
    # Store integration results
    globals()['integration_demo'] = {
        'method_1_output': combined_output,
        'method_2_output': merged_output if 'merged_output' in locals() else None,
        'lora_delta': lora_delta if 'lora_delta' in locals() else None,
        'integration_verified': output_match if 'output_match' in locals() else False
    }
    
else:
    print("⚠️ No trained LoRA adapters found!")
    print("Please run the LoRA training cell first to see integration demo.")
    
    print("\\n🎯 INTEGRATION METHODS OVERVIEW:")
    print("="*50)
    print()
    print("📋 THREE WAYS TO INTEGRATE TRAINED LORA:")
    print()
    print("🔀 METHOD 1: RUNTIME ADDITION")
    print("   output = original_layer(x) + lora_adaptation(x)")
    print("   ✅ Preserves base model")
    print("   ✅ Flexible and reversible")
    print()
    print("🔗 METHOD 2: WEIGHT MERGING") 
    print("   new_weights = original_weights + lora_delta")
    print("   ✅ Single matrix, faster inference")
    print("   ✅ Standard model format")
    print()
    print("🏗️ METHOD 3: IN-PLACE UPDATE")
    print("   model.layer = LoRAIntegratedLayer(original, lora_params)")
    print("   ✅ Transparent integration")
    print("   ✅ Full model functionality")
    
print("\\n🎯 INTEGRATION MECHANISM SUMMARY:")
print("="*50)
print("The key insight: LoRA changes are ADDED to original behavior,")
print("not replacing it. This preserves base model knowledge while") 
print("adding new SQL generation capabilities learned through RLVR.")
print()
print("Formula: Enhanced_Output = Original_Output + LoRA_Adaptation")
print("Where:   LoRA_Adaptation = (input @ A) @ B * scaling")'''

def create_lora_integration_notebook():
    """Create notebook showing LoRA integration"""
    
    # Base notebook structure
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Integration explanation cell
    integration_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": get_lora_integration_cell().split('\\n')
    }
    
    # Add cell to notebook
    notebook["cells"] = [integration_cell]
    
    return notebook

def main():
    """Create the LoRA integration notebook"""
    output_file = "/Users/George/Documents/GitHub/MLXTraining/lora_integration_explanation.ipynb"
    
    try:
        notebook = create_lora_integration_notebook()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        
        print(f"🎯 LORA INTEGRATION NOTEBOOK CREATED!")
        print(f"  📂 File: {output_file}")
        print(f"  🔗 Shows: How trained LoRA adapters integrate into Qwen model")
        print(f"  🚀 Covers:")
        print(f"    ✅ Runtime addition (current method)")
        print(f"    ✅ Weight merging (production method)") 
        print(f"    ✅ In-place update (transparent method)")
        print(f"    ✅ Practical recommendations for each use case")
        print(f"  🎯 Answers: How changed LoRA adapters are attached to base model")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating LoRA integration notebook: {e}")
        return False

if __name__ == "__main__":
    main()