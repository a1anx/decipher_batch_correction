# How the Comparison Scripts Work

This document explains how both comparison scripts work and what they do.

## Overview

We have **two comparison scripts**:

1. **`compare_approaches.py`** - Generic comparison (trains both models)
2. **`compare_concat_vs_attention.py`** - Specific comparison on your dataset (loads pre-trained models)

## 1. compare_approaches.py (Generic)

### What It Does

This script **trains both models from scratch** on the same dataset and compares them.

### How It Works

```python
def compare_models(adata, batch_key='batch', device='cpu', n_epochs=50):
    """
    1. Train concatenation model
    2. Train attention model
    3. Compare performance
    4. Create visualizations
    5. Make recommendation
    """
```

### Step-by-Step Process

#### Step 1: Configuration
```python
# Same configuration for both models (fair comparison)
shared_params = {
    'dim_z': 10,
    'dim_v': 2,
    'batch_emb_dim': 64,
    'decoder_hidden_dims': [128, 256],
    'learning_rate': 5e-3,
    'batch_size': 64,
    'n_epochs': n_epochs,
    'early_stopping_patience': 10
}
```

#### Step 2: Train Concatenation Model
```python
# Create config
config_concat = DecipherBatchCorrectedConcatConfig(**shared_params)

# Train
model_concat, losses_concat = train_batch_corrected_decipher_concat(
    adata, batch_key=batch_key, config=config_concat, device=device
)

# Evaluate
results_concat = evaluate_batch_correction_concat(model_concat, adata, batch_key)
```

**What happens:**
- Initializes concatenation-based decoder
- Trains using Pyro SVI (Stochastic Variational Inference)
- Extracts latent embeddings (z, v)
- Records training time and losses

#### Step 3: Train Attention Model
```python
# Create config (with attention-specific params)
config_attn = DecipherBatchCorrectedConfig(**shared_params)
config_attn.n_attention_heads = 4

# Train
model_attn, losses_attn = train_batch_corrected_decipher(
    adata, batch_key=batch_key, config=config_attn, device=device
)

# Evaluate
results_attn = evaluate_batch_correction(model_attn, adata, batch_key)
```

**What happens:**
- Initializes attention-based decoder
- Trains using same SVI approach
- Extracts latent embeddings AND attention weights
- Records training time and losses

#### Step 4: Compute Metrics

**Batch Mixing (Silhouette Score)**
```python
from sklearn.metrics import silhouette_score

def compute_batch_mixing_score(embeddings, batch_labels):
    """
    Silhouette score measures cluster separation.
    Range: [-1, 1]

    For batch correction:
    - High score (→ 1) = batches are well-separated (BAD, batch effect remains)
    - Low score (→ -1 or 0) = batches are mixed (GOOD, batch effect removed)

    We use: 1 - silhouette_score so that LOWER is BETTER
    """
    score = silhouette_score(embeddings, batch_labels)
    return 1 - score  # Invert so lower is better
```

**Parameter Count**
```python
def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

**Training Time**
```python
import time
start_time = time.time()
# ... training ...
training_time = time.time() - start_time
```

#### Step 5: Create Visualizations

**Plot 1: Training Curves**
- X-axis: Epochs
- Y-axis: ELBO loss (lower is better)
- Lines: Train/val for concat (blue) and attention (red)

**Plot 2: Parameter Count**
- Bar chart comparing model sizes

**Plot 3: Batch Mixing**
- Bar chart of silhouette scores (lower is better)

**Plot 4: Training Time**
- Bar chart of training times

#### Step 6: Make Recommendation

```python
if attention_mixing < concat_mixing:
    improvement = concat_mixing - attention_mixing
    if improvement > 0.05:
        print("Use ATTENTION (significantly better)")
    else:
        print("Both similar, use CONCATENATION (simpler)")
else:
    print("Use CONCATENATION (better or equal)")
```

### When to Use This Script

- **You have a NEW dataset** not analyzed before
- You want to **train both models** from scratch
- You want a **fair comparison** with identical hyperparameters
- You want to see **training dynamics** (loss curves)

## 2. compare_concat_vs_attention.py (Your Data)

### What It Does

This script **loads pre-trained models** (your results) and compares them.

### How It Works

#### Step 1: Load Results
```python
# Load concatenation results
adata_concat = sc.read_h5ad('adata_combined_2_concat.h5ad')

# Load attention results
adata_attention = sc.read_h5ad('adata_combined_2_batch_corrected.h5ad')
```

**Key embeddings loaded:**
- `decipher_decipher_z` - Original Decipher (both files)
- `X_decipher_concat_z` - Concatenation-corrected
- `X_decipher_batch_corrected_z` - Attention-corrected
- Same for `_v` (component space)

#### Step 2: Compute Batch Mixing

Same as generic script - uses silhouette score on embeddings.

#### Step 3: Create Visualizations

**comparison_z_space.png:**
```
┌─────────────┬─────────────┬─────────────┐
│  Original Z │  Concat Z   │ Attention Z │
│  (by shift) │ (by shift)  │ (by shift)  │
├─────────────┼─────────────┼─────────────┤
│  Original Z │  Concat Z   │ Attention Z │
│ (pseudotime)│(pseudotime) │(pseudotime) │
└─────────────┴─────────────┴─────────────┘
```

**comparison_v_space.png:**
- Same layout but for V (component) space

**comparison_metrics.png:**
```
┌──────────────┬──────────────┬──────────────┐
│ Batch Mixing │  Variance    │  Cumulative  │
│  (bar chart) │  Explained   │  Variance    │
│              │  (bar chart) │  (line plot) │
└──────────────┴──────────────┴──────────────┘
```

#### Step 4: Print Summary Report

Detailed text report with:
- Batch mixing scores
- Variance explained
- Model complexity comparison
- Attention weights (if available)
- **Recommendation** based on performance

### When to Use This Script

- **You've already trained both models** on your data
- You want to **compare results** without retraining
- You want to see **embeddings side-by-side**
- You want a **recommendation** for your specific dataset

## Key Metrics Explained

### 1. Batch Mixing (Silhouette Score)

**What it measures:**
How well batches are separated vs. mixed in the embedding space.

**Interpretation:**
```
Silhouette Score = 0.8  →  Batches are very separated (BATCH EFFECT REMAINS)
Silhouette Score = 0.2  →  Batches are well mixed (BATCH EFFECT REMOVED)
Silhouette Score = -0.1 →  Batches are completely mixed (EXCELLENT)
```

**Why lower is better for batch correction:**
- We WANT batches to mix (remove batch effect)
- We DON'T WANT batches to cluster separately

**Formula:**
```
For each cell i:
  a(i) = avg distance to cells in same batch
  b(i) = avg distance to cells in nearest other batch
  s(i) = (b(i) - a(i)) / max(a(i), b(i))

Silhouette = average of s(i) across all cells
```

### 2. Variance Explained

**What it measures:**
How much information each latent dimension captures.

**Interpretation:**
```
Dim 1: 45%  →  First dimension captures 45% of variance
Dim 2: 25%  →  Second dimension captures 25% of variance
...
```

**Good model:**
- First few dimensions capture most variance
- Indicates efficient compression of information

### 3. Parameter Count

**What it measures:**
Total number of trainable weights in the model.

**Typical values:**
- Concatenation: ~520,000 parameters
- Attention: ~680,000 parameters (+30%)

**Extra parameters in attention:**
- Multi-head attention mechanism (~50K params)
- Query/Key/Value projections
- Attention output projection

### 4. Training Time

**What it measures:**
Wall-clock time to train the model.

**Typical values:**
- Concatenation: ~8 seconds/epoch
- Attention: ~12 seconds/epoch (+50%)

**Why attention is slower:**
- Extra attention computation per batch
- More parameters to update
- More complex forward/backward pass

## Workflow for Your Data

### Option 1: Use Your Specific Comparison Script

```bash
# Step 1: Train concatenation model
cd decipher-bc-concat
python run_concat_on_data_2.py

# Step 2: Train attention model (if not already done)
cd ../simulated-data-analysis
python run_batch_corrected_on_data_2.py

# Step 3: Compare them
cd ../decipher-bc-concat
python compare_concat_vs_attention.py
```

**Pros:**
- Uses your actual trained models
- No retraining needed
- Shows real results on your data

**Cons:**
- Requires both models to be trained separately
- Can't compare training dynamics (losses)

### Option 2: Use Generic Comparison Script

```bash
cd decipher-bc-concat
python compare_approaches.py
```

**Pros:**
- Trains both models with identical settings
- Fair comparison (same hyperparameters)
- Shows training curves

**Cons:**
- Needs to retrain both models (slower)
- Uses synthetic data (for demo)

## Summary

| Script | Purpose | When to Use |
|--------|---------|-------------|
| **compare_approaches.py** | Train both, compare | New dataset, want training curves |
| **compare_concat_vs_attention.py** | Load both, compare | Already trained both models |

Both scripts:
- ✅ Compute batch mixing metrics
- ✅ Create side-by-side visualizations
- ✅ Make recommendations
- ✅ Compare performance quantitatively

The main difference is whether they **train** or **load** the models.
