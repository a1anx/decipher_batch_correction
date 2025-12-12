# Testing Guide: Concatenation Approach on Your Data

This guide shows you how to test the concatenation approach on the same data as the attention approach and get identical visualizations.

## Your Data: adata_combined_2.h5ad

**Location**: `../simulated-data-analysis/simulation/adata/adata_combined_2.h5ad`

**Properties**:
- Batch column: `'shift'`
- 4 conditions: `'False'`, `'Alpha 0.01'`, `'Alpha 0.05'`, `'Alpha 0.1'`
- Pseudotime column: `'latent_t'`
- Branch column: `'branch_id'`

## Quick Start

```bash
cd decipher-bc-concat

# Step 1: Train concatenation model on your data
python run_concat_on_data_2.py

# Step 2: Visualize all results (includes all visualizations)
python visualize_all_concat_results.py

# Step 3: Compare with attention approach
python compare_concat_vs_attention.py
```

## Detailed Workflow

### Step 1: Train Concatenation Model

**Script**: `run_concat_on_data_2.py`

**What it does**:
1. Loads `adata_combined_2.h5ad`
2. Configures concatenation model (same hyperparameters as attention)
3. Trains using Pyro SVI
4. Extracts embeddings (z and v)
5. Saves to `adata_combined_2_concat.h5ad`

**Run**:
```bash
python run_concat_on_data_2.py
```

**Expected output**:
```
==================================================
CONCATENATION-BASED BATCH CORRECTION
==================================================
This is the SIMPLER baseline approach

[1/5] Loading data...
✓ Loaded data: 30000 cells, 2000 genes
  Batch column: 'shift' with 4 conditions

[2/5] Configuring model...
✓ Configuration initialized
  - Latent z dimension: 10
  - Approach: CONCATENATION

[3/5] Training model...
✓ Training complete!
  Final validation loss: 2450.12

[4/5] Extracting embeddings...
✓ Embeddings extracted

[5/5] Saving results...
✓ Results saved

OUTPUTS SAVED:
  - X_decipher_concat_z (10D)
  - X_decipher_concat_v (2D)
```

**Output file**: `../simulated-data-analysis/simulation/adata/adata_combined_2_concat.h5ad`

### Step 2: Visualize All Results

**Script**: `visualize_all_concat_results.py`

**What it does**:
- Combines `visualize_batch_correction_2.py` AND `visualize_v_space_by_pseudotime.py`
- Creates all visualizations in one run
- 3×3 method comparison grid
- Biological variables (time, branch)
- V space colored by condition AND pseudotime
- Computes batch mixing metrics

**Run**:
```bash
python visualize_all_concat_results.py
```

**Output files**:
1. `concat_method_comparison.png` - 3×3 grid comparing methods
2. `concat_biological_variables.png` - Time, branch, variance
3. `concat_v_space_four_panel.png` - V space comprehensive view
4. `concat_v_original_comparison.png` - Original V (condition vs pseudotime)
5. `concat_v_batch_corrected_comparison.png` - Concat V (condition vs pseudotime)

**What to look for**:
- ✅ Mixed colors in concat plots = good batch correction
- ✅ Biological structure (time, branch) preserved
- ✅ Lower silhouette score = better mixing
- ✅ Smooth pseudotime gradient = biological trajectory preserved
- ✅ Less color separation by condition = batch effect removed

### Step 3: Compare with Attention

**Script**: `compare_concat_vs_attention.py`

**What it does**:
- Loads BOTH concatenation and attention results
- Computes batch mixing for both
- Creates side-by-side comparisons
- Makes recommendation

**Prerequisites**:
```bash
# Make sure attention model was trained
cd ../simulated-data-analysis
python run_batch_corrected_on_data_2.py  # If not already done
cd ../decipher-bc-concat
```

**Run**:
```bash
python compare_concat_vs_attention.py
```

**Output files**:
1. `comparison_z_space.png` - Z latent space comparison
2. `comparison_v_space.png` - V component space comparison
3. `comparison_metrics.png` - Quantitative metrics

**Output report**:
```
COMPREHENSIVE COMPARISON SUMMARY
================================================================================

1. BATCH MIXING (Silhouette Score - lower is better)
--------------------------------------------------------------------------------
  Original:           0.6543
  Concatenation:      0.2134  (improvement: +0.4409)
  Attention:          0.1987  (improvement: +0.4556)

  Winner: ATTENTION (better by 0.0147)

2. VARIANCE EXPLAINED (Top 3 dimensions)
  Concatenation: Dim 1: 45.2%, Dim 2: 18.3%, Dim 3: 9.1%
  Attention:     Dim 1: 43.8%, Dim 2: 19.1%, Dim 3: 10.2%

5. RECOMMENDATION
--------------------------------------------------------------------------------
  ⚖ BOTH APPROACHES WORK WELL
    - Performance difference is small (0.0147)
    - Use CONCATENATION for simplicity
    - Use ATTENTION if you need interpretability (attention weights)
```

## Files Created (Summary)

After running all steps, you'll have:

### From Training (`run_concat_on_data_2.py`)
- `../simulated-data-analysis/simulation/adata/adata_combined_2_concat.h5ad`

### From Visualization (`visualize_all_concat_results.py`)
- `concat_method_comparison.png`
- `concat_biological_variables.png`
- `concat_v_space_four_panel.png`
- `concat_v_original_comparison.png`
- `concat_v_batch_corrected_comparison.png`

### From Comparison (`compare_concat_vs_attention.py`)
- `comparison_z_space.png`
- `comparison_v_space.png`
- `comparison_metrics.png`

## Interpreting Results

### Good Batch Correction Should Show:

1. **Batch Mixing (Quantitative)**
   - Lower silhouette score compared to original
   - Ideally < 0.3 (good mixing)
   - < 0.1 (excellent mixing)

2. **Visual Mixing (Qualitative)**
   - Colors (batches) are intermingled in plots
   - Not clustered by condition
   - BUT biological structure (time, branch) still visible

3. **Biological Preservation**
   - Pseudotime gradient smooth and continuous
   - Branch structure maintained
   - Known biological relationships preserved

### Red Flags:

❌ **Over-correction**:
- Biological structure lost
- No clear pseudotime progression
- Branch IDs completely mixed

❌ **Under-correction**:
- Batches still clearly separated
- High silhouette score (> original)
- Condition dominates over biology

❌ **Artifacts**:
- Strange clustering patterns
- Discontinuities in pseudotime
- Loss of known cell types

## Comparison with Attention Results

You can directly compare files:

| Concatenation | Attention |
|---------------|-----------|
| `concat_method_comparison.png` | `../simulated-data-analysis/batch_correction_comparison_2.png` |
| `concat_v_space_four_panel.png` | `../simulated-data-analysis/v_*_comparison*.png` |

**Key Questions:**
1. Is concatenation "good enough"?
2. Does attention provide meaningful improvement?
3. Is the improvement worth the added complexity?

## Expected Results

Based on typical performance:

### If Your Data Has Simple Batch Effects:
- Concatenation should perform **similarly** to attention
- Silhouette scores within 0.05 of each other
- **Recommendation**: Use concatenation (simpler)

### If Your Data Has Complex Batch Effects:
- Attention should perform **better** than concatenation
- Silhouette improvement > 0.05
- **Recommendation**: Use attention (worth the complexity)

### For adata_combined_2.h5ad Specifically:
- This is simulated data with controlled batch effects
- Both approaches should work well
- Expect small difference (~0.01-0.02 in silhouette)
- Concatenation likely sufficient

## Troubleshooting

### Error: File not found
```
⚠ ERROR: Concatenation results not found!
```
**Solution**: Run `python run_concat_on_data_2.py` first

### Error: Different shapes
```
AssertionError: Datasets have different shapes!
```
**Solution**: Make sure both models were trained on the same dataset

### Warning: No pseudotime column
```
⚠ Warning: 'latent_t' column not found
```
**Solution**: Check your data has the `latent_t` column

### Poor Performance
If batch mixing is worse than original:
1. Check training completed (no early stopping too early)
2. Try increasing `batch_emb_dim` (32 → 64)
3. Try more training epochs
4. Check data quality

## Next Steps

After testing:

1. **If concatenation works well**: Use it! It's simpler.
2. **If you need better results**: Try attention approach
3. **For production**: Consider both and pick based on your needs
4. **For research**: Use attention for interpretability

## Questions?

Refer to:
- `README.md` - Full concatenation documentation
- `HOW_COMPARE_WORKS.md` - How comparison scripts work
- `ARCHITECTURE_COMPARISON.md` - Visual comparison of architectures
- `QUICKSTART.md` - Quick reference
