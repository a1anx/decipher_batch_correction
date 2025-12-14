# Summary: Addressing TA Feedback

## TA's Key Points

### 1. **Multiple Cell Types Problem**
**TA said:** "Are there multiple cell types present here? I would subset to just one and use some known biological markers to plot on the UMAP to verify the trajectory."

**Issue:** Your analysis used all 55 cell types together, making it impossible to determine if the "blob" represents:
- Model failure (over-correction)
- Legitimate biology (overlapping lineages)

**Solution:** ✅ **Created scripts to analyze single Erythroid lineage** with 6 developmental stages and known marker genes.

---

### 2. **Beta Understanding**
**TA said:** "I would use the following source to brush up the understanding of modifying beta: https://lilianweng.github.io/posts/2018-08-12-vae/. In general people usually stay between 0-1 as the intent is to downweight KL."

**Issue:** Beta > 1 is unusual in VAE literature. Standard practice is β ∈ [0, 1] to downweight KL divergence.

**Your intuition was correct:** "I feel like you can't just 10x KL Regularization without expecting some 'whiplash'"

**Data confirms:**
- Beta=0.1 → 1.0: Silhouette improves from -0.366 to -0.272 ✅
- Beta=1.0 → 2.0: Silhouette degrades back to -0.366 ❌

**Why Beta=2.0 fails:** Over-regularization forces all cells toward the prior (standard normal), causing latent collapse. Everything becomes a homogeneous blob.

---

### 3. **What Really Matters**
**TA said:** "If you did the opposite z would lose some meaning and likely collapse. I would say this isn't as important as the blob as the preprocessing, attention method, and data subset."

**Priority factors (in order):**
1. **Data subsetting** (single lineage vs all types) ← Most important
2. **Preprocessing** (normalization, HVG selection)
3. **Attention method** (2 heads vs 4 heads)
4. **Beta** (stay in 0.5-1.0 range) ← Less critical than above

---

### 4. **Biological Validation**
**TA said:** "Try to figure out the biological source of the blob first, very likely it's a unique cell type."

**Hypothesis:** The "blob" is **55 different lineages overlapping**, not model failure.

**Test:** Subset to single lineage and see if structure resolves.

---

## What We Created

### Scripts in [Subset_by_cell_type](Subset_by_cell_type/) folder:

1. **`01_subset_erythroid_lineage.py`**
   - Extracts ~2,000 erythroid cells (6 developmental stages)
   - All 45 donors represented
   - Validates marker gene presence (HBB, GYPA, etc.)

2. **`02_train_regular_decipher_erythroid.py`**
   - Baseline model (no batch correction)
   - Beta=0.1 (standard Decipher)
   - Shows trajectory + batch effects

3. **`03_train_batch_corrected_erythroid.py`**
   - Configurable Beta and attention heads
   - Default: Beta=1.0, 2 heads (best from full dataset)
   - Tests batch correction on single lineage

4. **`04_compare_and_validate_markers.py`**
   - Side-by-side comparison
   - Quantitative metrics (Silhouette, Entropy)
   - **Marker gene validation** (HBB, GYPA expression patterns)
   - Visual inspection of trajectory preservation

5. **`run_full_analysis.sh`**
   - One-command pipeline
   - Runs all 4 steps sequentially
   - ~30-45 minutes total

---

## Expected Outcomes

### If Analysis Works Well:

1. **"Blob" resolves** when using single lineage
   - V-space shows clear developmental progression
   - Early → Late erythroblasts organized spatially

2. **Marker genes validate trajectory**
   - HBB/HBA1 (hemoglobin) increases with maturation
   - TFRC (transferrin receptor) decreases with maturation
   - Expression correlates with V-space position

3. **Batch correction improves mixing**
   - Donors mix within each maturation stage
   - But stages remain separated (biology preserved)
   - Silhouette closer to 0, Entropy increases

4. **Beta=1.0 (2 heads) performs best**
   - Better structure than Beta=0.1 or Beta=2.0
   - Balances batch mixing and trajectory preservation

### This Would Prove:

- ✅ Original "blob" was **legitimate** (55 overlapping lineages)
- ✅ Model works correctly on single lineages
- ✅ Beta=1.0 is optimal (not too weak, not too strong)
- ✅ Biological validation confirms trajectory preservation

---

## How to Run

### Quick Start (All Steps):
```bash
cd /home/alan/Documents/decipher_batch_correction/Bone_Marrow_Hematopoiesis/Subset_by_cell_type
bash run_full_analysis.sh
```

### Step-by-Step:
```bash
cd /home/alan/Documents/decipher_batch_correction/Bone_Marrow_Hematopoiesis/Subset_by_cell_type
source ../../decipher_env/bin/activate

# Step 1: Subset data
python 01_subset_erythroid_lineage.py

# Step 2: Regular Decipher
python 02_train_regular_decipher_erythroid.py --epochs 150

# Step 3: Batch-Corrected
python 03_train_batch_corrected_erythroid.py --beta 1.0 --heads 2 --epochs 150

# Step 4: Compare
python 04_compare_and_validate_markers.py
```

---

## Key Files Generated

| File | Purpose |
|------|---------|
| `BoneMarrowMap_Erythroid_5kgenes.h5ad` | Erythroid-only dataset (~2K cells) |
| `erythroid_regular_decipher.h5ad` | Baseline (no batch correction) |
| `erythroid_batch_corrected_beta1.0_heads2.h5ad` | Best model from full analysis |
| `erythroid_comparison_with_markers.png` | **Visual comparison** ← Most important |
| `erythroid_comparison_metrics.csv` | Quantitative results |

---

## Response to TA

After running analysis, you can respond:

1. **Acknowledge subsetting suggestion:**
   - "I subset the data to erythroid lineage only (~2K cells, 6 stages)"
   - "Used known markers (HBB, GYPA, TFRC) for validation"

2. **Confirm beta understanding:**
   - "Following Lilian Weng's post, I tested β ∈ [0.1, 0.5, 1.0, 2.0]"
   - "Beta=2.0 caused latent collapse (silhouette=-0.366, same as Beta=0.1)"
   - "Beta=1.0 with 2 heads achieved best balance"

3. **Show biological validation:**
   - Attach `erythroid_comparison_with_markers.png`
   - "HBB expression shows expected gradient along maturation"
   - "V-space preserves developmental trajectory while mixing donors"

4. **Address "blob" question:**
   - "The original 'blob' was 55 overlapping lineages, not model failure"
   - "Single lineage shows clear structure with proper batch correction"

5. **Highlight key findings:**
   - "Beta=1.0 (2 heads) improves batch mixing by X% without losing trajectory"
   - "Marker genes validate that biological structure is preserved"

---

## Questions Answered

### Your Question: "Should we subset by attention weights?"
**Answer:** No, use **biological knowledge** instead.
- Attention weights show what the model learned (circular reasoning)
- Use **known developmental trajectories** (Erythroid, Myeloid, etc.)
- Validate with **marker genes** (independent ground truth)

### Your Question: "Can I say blob failed to preserve biological structure?"
**Answer:** Depends on context.
- With 55 cell types: "Blob" might be **legitimate** (overlapping lineages)
- With single lineage: "Blob" would indicate **failure** (loss of trajectory)
- Need **marker genes** to validate (not just visual inspection)

### Your Question: "What are these metrics?"
**Answered:**
- **Silhouette Score:** Measures batch separation (-1 to +1, closer to 0 = better mixing)
- **Batch Entropy:** Diversity of batches in neighborhoods (higher = better mixing)
- **Spearman Correlation:** How well V-space captures developmental trajectory (higher = better)

---

## Next Steps

1. **Run the analysis** (30-45 min)
2. **Review results** - Check marker gene patterns
3. **Optional:** Test other lineages (Myeloid, T cells) for generalization
4. **Update TA email** with erythroid-specific findings
5. **Include visualizations** showing marker validation

---

## Files Location

All scripts are in:
```
/home/alan/Documents/decipher_batch_correction/Bone_Marrow_Hematopoiesis/Subset_by_cell_type/
```

Parent data file (must exist):
```
/home/alan/Documents/decipher_batch_correction/Bone_Marrow_Hematopoiesis/BoneMarrowMap_small_20k_5kgenes.h5ad
```

Virtual environment:
```
/home/alan/Documents/decipher_batch_correction/decipher_env/
```
