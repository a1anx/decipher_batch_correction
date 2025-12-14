# Erythroid Lineage Analysis - Following TA Feedback

This folder contains scripts to analyze Decipher batch correction on a **single cell lineage** (Erythroid), following TA feedback.

## Background

The TA correctly pointed out that analyzing all 55 cell types together makes it impossible to determine if the "blob" shape in V-space represents:
- **Model failure** (over-correction collapsing structure)
- **Legitimate biology** (multiple overlapping lineages)

By subsetting to a **single lineage with known trajectory**, we can:
1. Validate the model using **biological marker genes**
2. Clearly assess if batch correction preserves developmental progression
3. Determine if the "blob" was due to mixing multiple unrelated cell types

---

## Why Erythroid Lineage?

**Perfect for validation:**
- ✅ **Clear developmental trajectory**: HSC → Pro-Erythroblast → Basophilic → Polychromatic → Orthochromatic → Mature RBC
- ✅ **Good cell counts**: ~2,000 cells across 6 stages
- ✅ **All 45 donors represented**: Strong statistical power for batch correction
- ✅ **Known marker genes**: HBB, HBA1, GYPA, KLF1 (increase with maturation)
- ✅ **Well-studied biology**: Can validate if V-space captures real differentiation

---

## Workflow

### Step 1: Subset Data to Erythroid Lineage
```bash
cd /home/alan/Documents/decipher_batch_correction/Bone_Marrow_Hematopoiesis/Subset_by_cell_type
source ../../decipher_env/bin/activate
python 01_subset_erythroid_lineage.py
```

**Output:**
- `BoneMarrowMap_Erythroid_5kgenes.h5ad` - ~2,000 erythroid cells
- `erythroid_subset_summary.png` - Overview of subset

**Cell types included:**
- BFU-E (early progenitor)
- CFU-E (committed progenitor)
- Pro-Erythroblast
- Basophilic Erythroblast
- Polychromatic Erythroblast
- Orthochromatic Erythroblast

---

### Step 2: Train Regular Decipher (Baseline)
```bash
python 02_train_regular_decipher_erythroid.py --epochs 150
```

**Output:**
- `erythroid_regular_decipher.h5ad` - Model with no batch correction
- `erythroid_regular_decipher_results.png` - V-space visualizations

**Expected:**
- V-space shows developmental trajectory (early → late stages)
- Potential batch effects (donors clustering separately)

---

### Step 3: Train Batch-Corrected Decipher
```bash
# Best model from full dataset analysis: Beta=1.0, 2 attention heads
python 03_train_batch_corrected_erythroid.py --beta 1.0 --heads 2 --epochs 150
```

**Output:**
- `erythroid_batch_corrected_beta1.0_heads2.h5ad` - Batch-corrected model
- `erythroid_batch_corrected_beta1.0_heads2_results.png` - V-space visualizations

**Expected:**
- Preserved developmental trajectory
- Improved donor mixing within each maturation stage
- Better structure than Beta=0.1 or Beta=2.0

**Optional:** Test other configurations
```bash
# Try Beta=0.5 for comparison
python 03_train_batch_corrected_erythroid.py --beta 0.5 --heads 2 --epochs 150

# Try Beta=0.1 (weak regularization)
python 03_train_batch_corrected_erythroid.py --beta 0.1 --heads 4 --epochs 150
```

---

### Step 4: Compare Models & Validate with Marker Genes
```bash
python 04_compare_and_validate_markers.py \
    --regular erythroid_regular_decipher.h5ad \
    --batch-corrected erythroid_batch_corrected_beta1.0_heads2.h5ad
```

**Output:**
- `erythroid_comparison_with_markers.png` - Side-by-side comparison
- `erythroid_comparison_metrics.csv` - Quantitative metrics

**Validation checks:**
1. **Batch mixing metrics** (Silhouette, Entropy)
2. **Marker gene expression** (HBB, GYPA, etc.)
3. **Visual inspection** - Does trajectory look biological?

---

## Key Questions to Answer

### 1. Does the "blob" resolve with single lineage?
- If yes: Original blob was overlapping lineages (not model failure)
- If no: Model needs further refinement

### 2. Do marker genes show expected patterns?
- **HBB, HBA1** (hemoglobin): Should **increase** from Pro-Ery → Orthochromatic
- **TFRC** (transferrin receptor): High in early, **decreases** with maturation
- **GYPA**: Should be present across all erythroid cells

### 3. Is developmental trajectory preserved?
- V-space should organize cells by maturation stage
- Clear progression: Early progenitors → Late erythroblasts
- NOT organized by donor

### 4. Does batch correction improve mixing without losing biology?
- Donors should mix **within** each maturation stage
- But stages should remain **separated** from each other

---

## Understanding Beta Values (From TA Feedback)

The TA referenced [Lilian Weng's VAE post](https://lilianweng.github.io/posts/2018-08-12-vae/):

**Standard practice: β ∈ [0, 1]**
- β < 1: **Downweight** KL divergence (prevent posterior collapse)
- β = 1: Standard VAE (equal weight to reconstruction and KL)
- **β > 1: UNUSUAL** - Upweights KL, forces z toward prior

**Our results confirm this:**
| Beta | V-space Silhouette | Interpretation |
|------|-------------------|----------------|
| 0.1  | -0.366 | Weak regularization → Poor structure |
| 0.5  | -0.273 | Good balance |
| **1.0** | **-0.272** | Best balance (with 2 heads) |
| 2.0  | -0.366 | **Too strong** → Latent collapse |

**Key insight:** Beta=2.0 has **identical** structure to Beta=0.1 (-0.366), just for different reasons:
- Beta=0.1: Under-regularized, batch effects dominate
- Beta=2.0: Over-regularized, everything collapses to prior

---

## TA's Priority Recommendations

1. ✅ **Subset to single lineage** - Done (Erythroid)
2. ✅ **Use biological markers** - HBB, GYPA, TFRC, KLF1
3. ✅ **Understand beta properly** - β ∈ [0, 1] is standard
4. ⚠️ **Other factors matter more than beta:**
   - Data preprocessing (normalization, HVG selection)
   - Attention mechanism (2 heads vs 4 heads)
   - Data quality (single lineage vs all types)

---

## Expected Timeline

- **Step 1** (Subset): ~2 minutes
- **Step 2** (Regular Decipher): ~10-20 minutes (depending on hardware)
- **Step 3** (Batch-Corrected): ~10-20 minutes
- **Step 4** (Comparison): ~2 minutes

**Total:** ~30-45 minutes for complete analysis

---

## Interpreting Results

### Good Signs:
- ✅ Batch silhouette **closer to 0** (better mixing)
- ✅ Batch entropy **increases** (more donor diversity in neighborhoods)
- ✅ V-space shows **clear trajectory** aligned with maturation stages
- ✅ Marker genes show **biological gradients** (not random patterns)
- ✅ Cell types remain **separated** (biology preserved)

### Warning Signs:
- ⚠️ V-space becomes **homogeneous blob** (all stages mixed)
- ⚠️ Marker genes show **no pattern** with V-space position
- ⚠️ Batch silhouette gets **worse** (more separated)
- ⚠️ Cell types **overlap** completely (structure lost)

---

## Next Steps After Analysis

Based on results, you can:

1. **If Erythroid works well:**
   - Try other lineages (Myeloid, T cells) to confirm generalization
   - Use this configuration for full dataset
   - Write up results for TA showing marker validation

2. **If Erythroid shows issues:**
   - Investigate preprocessing (normalization, scaling)
   - Try different attention configurations
   - Check data quality (outliers, doublets)

3. **For TA response:**
   - Show marker gene validation plots
   - Demonstrate biological trajectory preservation
   - Explain beta choice with evidence
   - Compare single-lineage vs full-dataset results

---

## Files Generated

```
Subset_by_cell_type/
├── README.md (this file)
├── 01_subset_erythroid_lineage.py
├── 02_train_regular_decipher_erythroid.py
├── 03_train_batch_corrected_erythroid.py
├── 04_compare_and_validate_markers.py
├── BoneMarrowMap_Erythroid_5kgenes.h5ad
├── erythroid_subset_summary.png
├── erythroid_regular_decipher.h5ad
├── erythroid_regular_decipher_results.png
├── erythroid_batch_corrected_beta1.0_heads2.h5ad
├── erythroid_batch_corrected_beta1.0_heads2_results.png
├── erythroid_comparison_with_markers.png
└── erythroid_comparison_metrics.csv
```

---

## Questions?

If you encounter issues:
1. Check that virtual environment is activated: `source ../../decipher_env/bin/activate`
2. Verify input files exist in parent directory
3. Check error messages for missing dependencies
4. Ensure sufficient memory (erythroid subset is small, should run on most machines)
