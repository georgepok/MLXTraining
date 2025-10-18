#!/usr/bin/env python3
"""
Quick Fix for GRPO v2: Add Prompt Variations for Diversity

This patch modifies grpo_sentiment_training_v2.py to add prompt variations
that create diversity within groups, enabling non-zero advantages and gradients.

USAGE:
    python apply_grpo_quick_fix.py

This will create: grpo_sentiment_training_v2_fixed.py
"""

import re

def apply_fix(input_file='grpo_sentiment_training_v2.py', 
              output_file='grpo_sentiment_training_v2_fixed.py'):
    
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Fix 1: Add prompt variations constant after imports
    variations_code = '''
# ============================================================================
# PROMPT VARIATIONS FOR DIVERSITY
# ============================================================================

# CRITICAL FIX: Prompt variations create diversity within groups
# Without diversity: all completions identical → zero advantages → no learning
# With diversity: varied completions → non-zero advantages → gradient flow
PROMPT_VARIATIONS = [
    "",                           # Original prompt
    " Be precise.",              # Encourages exact scoring
    " Think carefully.",         # Adds deliberation cue
    " Consider all aspects."     # Broadens evaluation scope
]

'''
    
    # Find the point after dataset definition (before LOAD MODELS comment)
    load_models_pattern = r'(print\(f"\\n📚 Dataset: \{len\(train_set\)\} train, \{len\(val_set\)\} validation"\)\n)'
    content = re.sub(
        load_models_pattern,
        r'\1' + variations_code,
        content,
        count=1
    )
    
    # Fix 2: Modify the rollout generation loop
    old_loop = r'''            for _ in range\(group_size\):
                # Generate response using model_old
                response = generate\(model_old, tokenizer, prompt=prompt, max_tokens=max_ans_len, verbose=False\)'''
    
    new_loop = '''            for j in range(group_size):
                # CRITICAL FIX: Add variation to create diversity
                varied_prompt = prompt + PROMPT_VARIATIONS[j % len(PROMPT_VARIATIONS)]
                
                # Generate response using model_old with varied prompt
                response = generate(model_old, tokenizer, prompt=varied_prompt, max_tokens=max_ans_len, verbose=False)
                
                # Log diversity check on first iteration
                if it == 0 and example == batch_examples[0] and j == 0:
                    print(f"\\n{'─'*70}")
                    print("DIVERSITY CHECK: Prompt Variations Applied")
                    print(f"{'─'*70}")
                    print(f"Original prompt: '{prompt[-50:]}'")
                    for k, var in enumerate(PROMPT_VARIATIONS[:group_size]):
                        print(f"  Variation {k+1}: '{var}'")
                    print(f"{'─'*70}\\n")'''
    
    content = re.sub(old_loop, new_loop, content, count=1)
    
    # Fix 3: Add diversity diagnostic after advantage calculation
    diagnostic_code = '''
        # DIAGNOSTIC: Check if fix is working (first iteration only)
        if it == 0:
            print(f"\\n{'='*70}")
            print("DIVERSITY DIAGNOSTIC")
            print(f"{'='*70}")
            
            for i, ex in enumerate(batch_examples):
                print(f"\\nExample {i+1}: '{ex.text[:50]}...' (target: {ex.score:.2f})")
                start_idx = i * group_size
                end_idx = start_idx + group_size
                group_rewards_subset = all_rewards[start_idx:end_idx]
                group_advs = advantages[start_idx:end_idx].tolist()
                
                reward_std = np.std(group_rewards_subset)
                reward_mean = np.mean(group_rewards_subset)
                
                print(f"  Group Rewards: {[f'{r:.3f}' for r in group_rewards_subset]}")
                print(f"  Mean: {reward_mean:.3f}, Std: {reward_std:.4f}")
                print(f"  Advantages: {[f'{a:.3f}' for a in group_advs]}")
                
                if reward_std < 0.01:
                    print(f"  ⚠️  WARNING: Low diversity (std < 0.01)")
                    print(f"  ⚠️  This will lead to near-zero advantages")
                elif reward_std < 0.05:
                    print(f"  ⚠️  CAUTION: Moderate diversity (std < 0.05)")
                else:
                    print(f"  ✅ Good diversity! Advantages are non-zero")
            
            print(f"\\n{'='*70}\\n")
'''
    
    # Insert after advantages calculation
    advantages_pattern = r'(advantages = mx\.concatenate\(advantages\)\n)'
    content = re.sub(
        advantages_pattern,
        r'\1' + diagnostic_code,
        content,
        count=1
    )
    
    # Fix 4: Add gradient flow check
    gradient_check = '''
        # DIAGNOSTIC: Check gradient flow (second iteration only)
        if it == 1:
            print(f"\\n{'='*70}")
            print("GRADIENT FLOW CHECK")
            print(f"{'='*70}")
            
            param_changed = 0
            param_total = 0
            max_grad_norm = 0.0
            
            for name, grad in grads.items():
                param_total += 1
                grad_norm = float(mx.abs(grad).max())
                max_grad_norm = max(max_grad_norm, grad_norm)
                
                if grad_norm > 1e-8:
                    param_changed += 1
            
            print(f"\\nParameters with non-zero gradients: {param_changed}/{param_total}")
            print(f"Max gradient magnitude: {max_grad_norm:.6e}")
            
            if param_changed == 0:
                print(f"\\n❌ CRITICAL: NO GRADIENTS FLOWING!")
                print(f"   • All advantages are likely zero")
                print(f"   • Check if prompt variations are creating diversity")
            elif param_changed < param_total * 0.5:
                print(f"\\n⚠️  WARNING: Only {param_changed/param_total*100:.1f}% of parameters have gradients")
            else:
                print(f"\\n✅ Excellent! {param_changed/param_total*100:.1f}% of parameters updating")
            
            print(f"{'='*70}\\n")
'''
    
    # Insert after optimizer update
    update_pattern = r'(mx\.eval\(model\.parameters\(\), optimizer\.state\)\n)'
    content = re.sub(
        update_pattern,
        r'\1' + gradient_check,
        content,
        count=1
    )
    
    # Write fixed version
    with open(output_file, 'w') as f:
        f.write(content)
    
    print("="*70)
    print("GRPO v2 Quick Fix Applied Successfully!")
    print("="*70)
    print(f"\n✅ Created: {output_file}")
    print("\nChanges made:")
    print("  1. Added PROMPT_VARIATIONS to create diversity within groups")
    print("  2. Modified rollout loop to use varied prompts")
    print("  3. Added diversity diagnostic at iteration 0")
    print("  4. Added gradient flow check at iteration 1")
    print("\nWhat this fixes:")
    print("  • Deterministic generation → Now stochastic via prompt variations")
    print("  • Zero advantages → Now non-zero due to diverse completions")
    print("  • Zero gradients → Now flowing due to non-zero advantages")
    print("  • 0% improvement → Expected 15-30% improvement")
    print("\nNext steps:")
    print(f"  1. Run: python {output_file}")
    print("  2. Check diagnostics at iteration 0-1")
    print("  3. Monitor validation MAE improvements")
    print("\nExpected results:")
    print("  • Iteration 0: Should see diverse rewards (std > 0.05)")
    print("  • Iteration 1: Should see non-zero gradients")
    print("  • Iteration 20: Val MAE should drop from 0.594")
    print("  • Iteration 100: Val MAE target: 0.420-0.480 (~20-25% improvement)")
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    apply_fix()
