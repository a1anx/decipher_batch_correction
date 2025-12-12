# Delta Shift Analysis

This folder contains adapted scripts for running concatenation-based batch correction on delta shift data.

## Data Source

These scripts are adapted to work with:
- **Input file**: `/home/alan/Documents/decipher_batch_correction/simulation/archive/oldadata_alpha/adata_combined_2_delta.h5ad`
- **Batch column**: `delta` (instead of `shift`)
- **Data structure**: 4000 cells × 50 genes with continuous delta values

## Key Adaptations from Original Scripts

### 1. Batch Column
- Original: `shift` column with categorical values ('False', 'Alpha 0.01', etc.)
- Adapted: `delta` column with continuous numerical values

### 2. Input/Output Paths
- Original input: `../../simulation/archive/old_adata/adata_combined_2.h5ad`
- New input: `/home/alan/Documents/decipher_batch_correction/simulation/archive/oldadata_alpha/adata_combined_2_delta.h5ad`
- Output: `adata_combined_2_delta_concat.h5ad` (in same directory as input)

### 3. Visualization Colors
- Original: Fixed color map for categorical conditions
- Adapted: Dynamic colormap (`tab10`) for delta values with legend showing `Delta=X.XXX`

### 4. Missing Data Handling
- Adapted to handle missing PCA/scVI embeddings gracefully
- Uses ground truth latent_z when available

## Files

### run_concat_on_data_delta.py
Trains the concatenation-based batch-corrected Decipher model on delta shift data.

**Usage:**
```bash
cd /home/alan/Documents/decipher_batch_correction/decipher-batch-correction/decipher-bc-concat/delta_shift_analysis
source /home/alan/Documents/decipher_batch_correction/decipher_env/bin/activate
python run_concat_on_data_delta.py
```

**Output:**
- Saves results to: `adata_combined_2_delta_concat.h5ad`
- Adds embeddings: `X_decipher_concat_z` and `X_decipher_concat_v`

### visualize_all_concat_results_delta.py
Creates comprehensive visualizations of batch correction results.

**Usage:**
```bash
python visualize_all_concat_results_delta.py
```

**Output files:**
1. `concat_method_comparison_delta.png` - 3×3 grid comparing methods
2. `concat_biological_variables_delta.png` - Biological variables visualization
3. `concat_v_space_four_panel_delta.png` - V space four-panel comparison
4. `concat_v_original_comparison_delta.png` - Original V space side-by-side
5. `concat_v_batch_corrected_comparison_delta.png` - Corrected V space side-by-side

### compare_concat_vs_attention_delta.py
Compares concatenation vs attention approaches (requires attention results).

**Usage:**
```bash
python compare_concat_vs_attention_delta.py
```

**Requirements:**
- Needs `adata_combined_2_delta_batch_corrected.h5ad` (attention-based results)
- This file must be generated separately using attention-based batch correction

**Output files:**
1. `comparison_z_space_delta.png` - Z space comparison
2. `comparison_v_space_delta.png` - V space comparison
3. `comparison_metrics_delta.png` - Quantitative metrics

## Running the Full Analysis

```bash
# 1. Activate environment
cd /home/alan/Documents/decipher_batch_correction/decipher-batch-correction/decipher-bc-concat/delta_shift_analysis
source /home/alan/Documents/decipher_batch_correction/decipher_env/bin/activate

# 2. Train concatenation model
python run_concat_on_data_delta.py

# 3. Visualize results
python visualize_all_concat_results_delta.py

# 4. Compare with attention (if available)
python compare_concat_vs_attention_delta.py
```

## Notes

- All scripts maintain the same parent folder structure
- Output figures are saved in the current working directory
- The scripts import from parent directory modules (decipher_batch_corrected_concat, train_batch_corrected_concat)
- Batch mixing is measured using silhouette score (lower = better mixing)
