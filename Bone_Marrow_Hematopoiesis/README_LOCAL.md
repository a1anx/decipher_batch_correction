# BoneMarrowMap Batch Correction - Local Machine Guide

This guide is for running the analysis on your **local machine** using subsampled datasets.

## Overview

The BoneMarrowMap dataset (263k cells, 45 donors) is too large for most local machines. We create two subsampled versions:

- **SMALL** (20k cells, ~440 cells/donor): Quick testing (~5-10 minutes training)
- **MEDIUM** (90k cells, ~2k cells/donor): Full analysis (~20-60 minutes training)

Both maintain:
- All 45 donors represented (preserves batch effects!)
- Cell type proportions within each donor
- Top 5k highly variable genes (preserves biological signal)

## Files Overview

### For Local Machine (Subset Versions):
1. **`preprocess_bonemarrowmap_subset.py`** - Creates small & medium subsets
2. **`run_bonemarrowmap_subset_analysis.py`** - Trains on subsets
3. **`visualize_bonemarrowmap_subset.py`** - Visualizes subset results

### For GCP/Cloud (Full Dataset):
4. **`preprocess_bonemarrowmap.py`** - Filters full 263k cells to 5k genes
5. **`run_bonemarrowmap_analysis.py`** - Trains on full dataset
6. **`visualize_bonemarrowmap.py`** - Visualizes full results

## Quick Start (Local Machine)

### Step 1: Preprocess Data

Creates both SMALL and MEDIUM subsets:

```bash
cd Bone_Marrow_Hematopoiesis
python preprocess_bonemarrowmap_subset.py
```

**Time:** ~5-15 minutes
**Output:**
- `BoneMarrowMap_small_20k_5kgenes.h5ad` (~10-20 MB)
- `BoneMarrowMap_medium_90k_5kgenes.h5ad` (~40-80 MB)
- `BoneMarrowMap_subset_preprocessing_summary.txt`

### Step 2: Train Model

**Option A: Quick test with SMALL subset**
```bash
python run_bonemarrowmap_subset_analysis.py --subset small
```
- **Time:** ~5-10 minutes (CPU), ~2-3 minutes (GPU)
- **Best for:** Testing, debugging, quick iteration

**Option B: Full analysis with MEDIUM subset** (recommended)
```bash
python run_bonemarrowmap_subset_analysis.py --subset medium
```
- **Time:** ~20-60 minutes (CPU), ~10-15 minutes (GPU)
- **Best for:** Final analysis, publication-quality results

**Output:**
- `bonemarrowmap_{subset}_batch_corrected.h5ad` - Results with embeddings
- `bonemarrowmap_{subset}_loss_curves.png` - Training progress

### Step 3: Visualize Results

```bash
python visualize_bonemarrowmap_subset.py --subset medium
```

**Output:**
- `bonemarrowmap_{subset}_batch_correction_comparison.png` - 3×3 grid
- `bonemarrowmap_{subset}_attention_analysis.png` - Attention patterns
- `bonemarrowmap_{subset}_biological_preservation.png` - Cell type purity

## Detailed Workflow

### 1. Data Preprocessing

```bash
python preprocess_bonemarrowmap_subset.py
```

**What it does:**
1. Loads full BoneMarrowMap dataset (3.5 GB) using memory-efficient backed mode
2. Explores metadata (donors, cell types, etc.)
3. Creates stratified subsamples:
   - SMALL: ~440 cells per donor × 45 donors = 20k cells
   - MEDIUM: ~2,000 cells per donor × 45 donors = 90k cells
4. Filters to 5k highly variable genes using Seurat v3 method
5. Saves compressed datasets ready for training

**Memory usage:** ~4-6 GB RAM during processing
**Disk space:** Original (3.5 GB) + Small (10-20 MB) + Medium (40-80 MB)

**Key settings:**
- Stratified sampling preserves cell type proportions within each donor
- HVG selection uses batch-aware method to preserve batch effects
- Random seed=42 for reproducibility

### 2. Model Training

```bash
# Small subset (quick test)
python run_bonemarrowmap_subset_analysis.py --subset small

# Medium subset (recommended)
python run_bonemarrowmap_subset_analysis.py --subset medium --epochs 150
```

**What it does:**
1. Loads preprocessed subset
2. Auto-detects batch/donor and cell type columns
3. Configures Decipher batch correction model:
   - 10D latent space (z)
   - 2D trajectory space (v)
   - 64D batch embeddings
   - 4-head attention mechanism
4. Trains with early stopping
5. Extracts embeddings, attention weights, UMAP
6. Saves results

**Configuration:**

| Setting | Small | Medium |
|---------|-------|--------|
| Cells | 20k | 90k |
| Batch size | 128 | 256 |
| Default epochs | 50 | 150 |
| Validation split | 10% | 10% |
| Time (CPU) | ~5-10 min | ~20-60 min |
| Time (GPU) | ~2-3 min | ~10-15 min |
| Memory | ~2-4 GB | ~4-8 GB |

**Optional arguments:**
- `--subset {small,medium}` - Which subset to use
- `--epochs N` - Override default number of epochs

### 3. Visualization

```bash
python visualize_bonemarrowmap_subset.py --subset medium
```

**Creates 3 comprehensive figures:**

#### Figure 1: Batch Correction Comparison (3×3 grid)
- **Row 1:** UMAP embeddings colored by batch, cell type, pseudotime
- **Row 2:** Latent Z space (first 2D) colored by batch, cell type, attention
- **Row 3:** Component V space (trajectory) colored by batch, cell type, pseudotime

#### Figure 2: Attention Analysis
- **Panel 1:** Histogram of attention strength across all cells
- **Panel 2:** Mean attention by batch/donor (top 15)
- **Panel 3:** Mean attention by cell type (top 12)

#### Figure 3: Biological Signal Preservation
- **Panel 1:** UMAP colored by cell type purity
- **Panel 2:** Distribution of cell type purity scores
- **Panel 3:** Cell type counts (top 15)
- **Panel 4:** Summary metrics (purity, batch mixing, attention)

**Metrics computed:**
- Silhouette score for batch mixing (lower = better)
- Cell type purity in k=30 neighborhoods (higher = better)
- Attention strength statistics

## System Requirements

### Minimum (SMALL subset):
- RAM: 4 GB
- Disk: 5 GB free
- Time: ~15 minutes total

### Recommended (MEDIUM subset):
- RAM: 8 GB
- Disk: 10 GB free
- CPU: 4+ cores
- Time: ~30-90 minutes total

### With GPU (optional):
- CUDA-capable GPU with 4+ GB VRAM
- Speeds up training 3-5x
- Install: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

## Interpreting Results

### Batch Correction Quality

**Good batch correction shows:**
- ✅ Well-mixed batches in UMAP (colors intermingled)
- ✅ Low silhouette score (<0.2)
- ✅ Similar attention across batches

**Poor batch correction shows:**
- ❌ Separated clusters by batch in UMAP
- ❌ High silhouette score (>0.5)
- ❌ Extreme attention differences

### Biological Signal Preservation

**Good preservation shows:**
- ✅ Cell types form distinct clusters in UMAP
- ✅ High cell type purity (>0.7)
- ✅ Clear trajectory structure in V space

**Poor preservation shows:**
- ❌ Cell types mixed together
- ❌ Low cell type purity (<0.5)
- ❌ No trajectory structure visible

### Attention Patterns

**Attention strength indicates:**
- **High attention** (>0.6): Strong batch effect, heavy correction applied
- **Medium attention** (0.3-0.6): Moderate batch effect
- **Low attention** (<0.3): Minimal batch effect, light correction

**Expected patterns:**
- Some donors may have consistently higher attention (technical artifacts)
- Some cell types may have higher attention (batch-sensitive types)

## Troubleshooting

### Out of Memory Error

**During preprocessing:**
```bash
# The backed mode should prevent this, but if it happens:
# Close other programs and try again
```

**During training:**
```bash
# Reduce batch size
python run_bonemarrowmap_subset_analysis.py --subset small  # Use smaller subset

# Or edit the script and reduce batch_size from 256 to 128 or 64
```

### CUDA Out of Memory

```bash
# Fallback to CPU (slower but works)
# Model will automatically detect and use CPU if GPU fails
```

### Missing Columns Error

```bash
# The script will list available columns
# Manually enter the correct column name when prompted
```

### Training is Very Slow

**Expected times (CPU):**
- SMALL: ~5-10 minutes
- MEDIUM: ~20-60 minutes

**If much slower:**
- Close other programs
- Check CPU usage (`top` or `htop`)
- Consider using SMALL subset for testing

### Results Look Strange

**Check:**
1. Training converged (loss curves stabilized)
2. Validation loss didn't increase dramatically (overfitting)
3. Attention weights are reasonable (0-1 range, not all zeros/ones)
4. UMAP computed successfully (check for errors in output)

## Comparing with Full Dataset (GCP)

The subset versions are designed to:
- ✅ Preserve batch effect patterns (all 45 donors)
- ✅ Maintain cell type diversity
- ✅ Show representative results
- ⚠️ May have slightly different UMAP structure (fewer cells)
- ⚠️ May have slightly lower statistical power

For publication-quality results with maximum statistical power, consider running the full 263k cell dataset on GCP (see `README.md`).

## Cost Comparison

| Approach | Hardware | Time | Cost |
|----------|----------|------|------|
| Local SMALL | Your machine | ~15 min | $0 |
| Local MEDIUM | Your machine | ~60 min | $0 |
| GCP Full (CPU) | n1-highmem-8 | ~4-6 hours | $2-3 |
| GCP Full (GPU) | n1-highmem-8 + T4 | ~1-2 hours | $1.50-2.50 |

## Next Steps

1. **Analyze results:**
   - Open PNG files to view visualizations
   - Check metrics in terminal output
   - Load `.h5ad` file in scanpy/jupyter for further analysis

2. **Compare subsets:**
   ```bash
   # Train both and compare
   python run_bonemarrowmap_subset_analysis.py --subset small
   python run_bonemarrowmap_subset_analysis.py --subset medium
   ```

3. **Try different parameters:**
   - Adjust number of epochs
   - Modify architecture in script (hidden layers, attention heads)
   - Change HVG count in preprocessing

4. **Scale to full dataset:**
   - If results look good, consider GCP for full 263k cells
   - See `README.md` for GCP instructions

## File Sizes (Approximate)

```
BoneMarrowMap_Annotated_Dataset_expandedFeatures.h5ad   3.5 GB (original, gitignored)
BoneMarrowMap_small_20k_5kgenes.h5ad                     15 MB (gitignored)
BoneMarrowMap_medium_90k_5kgenes.h5ad                    60 MB (gitignored)
bonemarrowmap_small_batch_corrected.h5ad                 20 MB (gitignored)
bonemarrowmap_medium_batch_corrected.h5ad                80 MB (gitignored)
*.png files                                               1-5 MB each (gitignored)
*.py scripts                                              10-15 KB each (tracked in git)
```

## References

- **BoneMarrowMap:** [GitHub](https://github.com/andygxzeng/BoneMarrowMap)
- **Publication:** Blood Cancer Discovery (2025)
- **Decipher:** Original trajectory inference method
- **This implementation:** Adds attention-based batch correction

## Support

If you encounter issues:
1. Check terminal output for error messages
2. Verify file names match exactly
3. Ensure you have enough RAM/disk space
4. Try SMALL subset first to test everything works
5. Check that dependencies are installed: `scanpy`, `torch`, `pyro-ppl`, `matplotlib`, `seaborn`, `scikit-learn`
