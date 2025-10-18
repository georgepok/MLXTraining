# Better RLVR + LoRA Demonstration Tasks

## Why SQL Failed
- 0.5B model doesn't have schema memorized
- Teaching new facts (table/column names) requires memorization, not learning patterns
- Too few examples (30) to teach entire database schema
- Reward signal was binary (executes or doesn't) with limited feedback

## Criteria for Good RLVR Demo

1. ✅ **Base model already has capability** - We refine, not teach from scratch
2. ✅ **Clear, measurable reward** - Easy to compute and unambiguous
3. ✅ **Pattern learning, not memorization** - Model learns rules/behavior
4. ✅ **Quick training** - See improvement in 50-200 steps
5. ✅ **Visible improvement** - Before/after difference is obvious

---

## Recommended Demonstrations

### 🏆 Option 1: Sentiment Intensity Calibration (BEST)

**Task:** Train model to output sentiment scores that match human ratings

**Why This Works:**
- Base model already understands sentiment
- Just needs to calibrate its intensity judgments
- Clear reward: difference between predicted and actual score
- Lots of training data available
- Shows actual learning of patterns

**Example:**
```
Text: "This movie was absolutely fantastic!"
Base model: "Positive (0.75)"  ← Too conservative
Target: "Positive (0.95)"
Reward: -0.20 (error penalty)

After training:
Model: "Positive (0.93)"  ← Much better!
Reward: -0.02
```

**Training Setup:**
- Dataset: Movie reviews with 1-5 star ratings
- Input: Review text
- Output: Sentiment + score (0-1)
- Reward: -|predicted_score - actual_score|
- Expected improvement: 0.3 → 0.1 average error (67% reduction)

**Advantages:**
- Natural RL task (continuous reward)
- Base model has strong priors
- Clear improvement metric
- Engaging demonstration

---

### 🎯 Option 2: Response Length Control

**Task:** Train model to generate responses of specific target length

**Why This Works:**
- Base model can generate text of any length
- Just needs to learn length control
- Reward is crystal clear: |actual_length - target_length|
- No domain knowledge required
- Shows direct cause-effect learning

**Example:**
```
Prompt: "Explain photosynthesis in exactly 50 words"
Base model: 73 words  → Reward: -23
After training: 51 words → Reward: -1
```

**Training Setup:**
- Dataset: Questions with target word counts
- Reward: -abs(generated_length - target_length)
- Expected improvement: Average error 20 words → 5 words

**Advantages:**
- Trivial reward computation
- Dramatic, visible improvement
- No ambiguity
- Fast training (model already can generate)

---

### 📝 Option 3: Markdown Format Compliance

**Task:** Train model to always use correct markdown formatting

**Why This Works:**
- Model knows markdown syntax
- Just needs consistent enforcement
- Binary reward: follows rules or doesn't
- Easy to verify
- Practical use case

**Example:**
```
Rules:
1. Headers must use # syntax
2. Code must be in ```blocks
3. Lists must use - or *

Base model (inconsistent):
"Here's how to do it:
Step 1: Do this
Step 2: Do that
here's the code: print('hello')"

After training (consistent):
"## How to Do It

- Step 1: Do this
- Step 2: Do that

```python
print('hello')
```"
```

**Reward Function:**
- +1 for each correct format element
- -1 for each violation
- Total normalized to 0-1

**Advantages:**
- Clear right/wrong
- Immediately visible
- Practical utility
- Shows behavioral refinement

---

### 🔢 Option 4: Math Problem Difficulty Calibration

**Task:** Train model to generate math problems of specific difficulty

**Why This Works:**
- Model can generate math problems
- Needs to learn difficulty = complexity
- Reward: User success rate on problems
- Shows sophisticated pattern learning

**Example:**
```
Target: Generate "easy" addition problems

Base model: "Calculate: 847 + 923 + 1092"  → Too hard (20% solve rate)
After training: "Calculate: 12 + 5"  → Perfect (90% solve rate)

Reward based on actual user performance
```

**Training Setup:**
- Generate problems at different difficulty levels
- Test on humans or use heuristics (digit count, operations)
- Reward = actual_difficulty matches target_difficulty

---

### 🎨 Option 5: Toxicity Avoidance (Safety Alignment)

**Task:** Reduce toxic/inappropriate responses while maintaining helpfulness

**Why This Works:**
- Base model occasionally generates problematic text
- RLVR perfect use case (safety reward)
- High practical value
- Clear improvement metric

**Reward:**
- +1.0 for helpful, safe response
- +0.5 for safe but not helpful
- -1.0 for toxic/unsafe
- 0 for unhelpful but safe

**Advantages:**
- Important real-world application
- Shows RL improving safety
- Measurable via toxicity classifiers

---

## Recommendation: Go with Option 1 (Sentiment Calibration)

**Why Sentiment is Best:**

1. **Perfect for RLVR:**
   - Continuous reward signal (not binary)
   - Clear notion of "better" (lower error)
   - No ambiguity in correctness

2. **Shows Real Learning:**
   - Model learns to calibrate judgments
   - Not just memorization
   - Generalizes to new examples

3. **Easy to Implement:**
   - Datasets readily available (IMDb, Yelp)
   - Simple reward: -|predicted - actual|
   - Quick to train (100-200 steps)

4. **Compelling Demo:**
   - Before/after is dramatic
   - Can show calibration curves
   - Intuitive to understand

5. **Actually Feasible:**
   - 0.5B model CAN do this
   - Not asking it to memorize facts
   - Just refining existing capability

---

## Implementation Plan: Sentiment RLVR

### Dataset
Use IMDb movie reviews:
- 50,000 reviews with ratings (1-10 stars)
- Convert to 0-1 scale
- Split: 40k train, 10k validation

### Training
```python
def sentiment_reward(generated_score, actual_score):
    error = abs(generated_score - actual_score)
    reward = 1.0 - error  # 0 to 1, higher is better
    return reward

# Example
text = "This movie was amazing!"
actual_score = 0.9

# Model generates: "Positive: 0.75"
predicted_score = 0.75
reward = sentiment_reward(0.75, 0.9)  # = 0.85

# After training: "Positive: 0.88"
reward = sentiment_reward(0.88, 0.9)  # = 0.98 (better!)
```

### Expected Results
- **Before training:** MAE = 0.25 (average error)
- **After 200 steps:** MAE = 0.12 (52% improvement)
- **Validation:** Generalizes to unseen reviews

### Success Criteria
✅ Average error decreases by 40%+
✅ Validation error also decreases (not overfitting)
✅ Can show calibration curve improvement
✅ Demonstrates RLVR + LoRA actually working

---

## Next Steps

1. Download IMDb dataset
2. Implement sentiment scoring prompt
3. Create reward function
4. Apply LoRA to Qwen2.5-0.5B
5. Train for 200 steps with real gradients
6. Evaluate improvement

This will be a **genuine demonstration of RLVR + LoRA working**, not just infrastructure.
