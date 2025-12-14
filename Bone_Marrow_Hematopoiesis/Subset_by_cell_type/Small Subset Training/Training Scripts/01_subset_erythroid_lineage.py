"""
Subset BoneMarrowMap dataset to Erythroid lineage only.

Following TA feedback:
- Focus on single lineage with clear developmental trajectory
- Erythroid differentiation: HSC → Pro-Ery → Basophilic → Polychromatic → Orthochromatic
- This allows validation using known marker genes (HBB, HBA1, GYPA, KLF1)

Output:
- Erythroid-only dataset for training Decipher models
- Preserves all donor (batch) information for batch correction
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("SUBSETTING BONEMARROWMAP TO ERYTHROID LINEAGE")
print("="*80)

# Load full dataset
print("\n1. Loading full dataset...")
adata = sc.read_h5ad("../BoneMarrowMap_small_20k_5kgenes.h5ad")
print(f"   Full dataset: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
print(f"   Cell types: {adata.obs['CellType'].nunique()}")
print(f"   Donors: {adata.obs['Donor'].nunique()}")

# Define erythroid lineage cell types
erythroid_types = [
    'Pro-Erythroblast',
    'Basophilic Erythroblast',
    'Polychromatic Erythroblast',
    'Orthochromatic Erythroblast',
    'BFU-E',  # Burst-forming unit erythroid (early progenitor)
    'CFU-E',  # Colony-forming unit erythroid (committed progenitor)
]

print(f"\n2. Subsetting to Erythroid lineage...")
print(f"   Target cell types: {len(erythroid_types)}")
for ct in erythroid_types:
    count = (adata.obs['CellType'] == ct).sum()
    n_donors = adata.obs[adata.obs['CellType'] == ct]['Donor'].nunique()
    print(f"     - {ct:35s}: {count:4d} cells, {n_donors:2d} donors")

# Subset to erythroid cells
adata_ery = adata[adata.obs['CellType'].isin(erythroid_types)].copy()

print(f"\n3. Erythroid subset statistics:")
print(f"   Total cells: {adata_ery.n_obs:,}")
print(f"   Cell types: {adata_ery.obs['CellType'].nunique()}")
print(f"   Donors: {adata_ery.obs['Donor'].nunique()}")
print(f"   Genes: {adata_ery.n_vars:,}")

# Check donor distribution
donor_counts = adata_ery.obs.groupby('CellType')['Donor'].nunique()
print(f"\n4. Donor coverage per cell type:")
for ct, n_donors in donor_counts.items():
    print(f"     {ct:35s}: {n_donors:2d}/{adata_ery.obs['Donor'].nunique()} donors")

# Verify marker genes are present
marker_genes = ['HBB', 'HBA1', 'HBA2', 'GYPA', 'KLF1', 'GATA1', 'TFRC', 'EPOR']
print(f"\n5. Checking erythroid marker genes:")
present_markers = [g for g in marker_genes if g in adata_ery.var_names]
missing_markers = [g for g in marker_genes if g not in adata_ery.var_names]

print(f"   Present ({len(present_markers)}/{len(marker_genes)}):")
for gene in present_markers:
    print(f"     ✓ {gene}")
if missing_markers:
    print(f"   Missing:")
    for gene in missing_markers:
        print(f"     ✗ {gene}")

# Save erythroid subset
output_file = "BoneMarrowMap_Erythroid_5kgenes.h5ad"
print(f"\n6. Saving erythroid subset...")
adata_ery.write(output_file)
print(f"   Saved to: {output_file}")

# Create summary visualization
print(f"\n7. Creating summary visualization...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Cell type distribution
ax = axes[0, 0]
celltype_counts = adata_ery.obs['CellType'].value_counts()
celltype_counts.plot(kind='barh', ax=ax, color='steelblue')
ax.set_xlabel('Number of Cells', fontsize=11)
ax.set_ylabel('Cell Type', fontsize=11)
ax.set_title('Erythroid Cell Type Distribution', fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Donor distribution
ax = axes[0, 1]
donor_counts_overall = adata_ery.obs['Donor'].value_counts()
ax.hist(donor_counts_overall.values, bins=20, color='coral', edgecolor='black')
ax.set_xlabel('Cells per Donor', fontsize=11)
ax.set_ylabel('Number of Donors', fontsize=11)
ax.set_title('Donor Representation', fontsize=12, fontweight='bold')
ax.axvline(donor_counts_overall.mean(), color='red', linestyle='--',
           label=f'Mean: {donor_counts_overall.mean():.1f}')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Cell type × Donor heatmap
ax = axes[1, 0]
crosstab = pd.crosstab(adata_ery.obs['CellType'], adata_ery.obs['Donor'])
# Show only subset of donors for readability
donor_subset = donor_counts_overall.nlargest(20).index
crosstab_subset = crosstab[donor_subset]
sns.heatmap(crosstab_subset, ax=ax, cmap='YlOrRd', cbar_kws={'label': 'Cell Count'},
            linewidths=0.5, linecolor='gray')
ax.set_xlabel('Donor (Top 20)', fontsize=11)
ax.set_ylabel('Cell Type', fontsize=11)
ax.set_title('Cell Type × Donor Distribution', fontsize=12, fontweight='bold')

# Summary statistics
ax = axes[1, 1]
ax.axis('off')
summary_text = f"""
ERYTHROID LINEAGE SUBSET SUMMARY

Total Cells: {adata_ery.n_obs:,}
Total Genes: {adata_ery.n_vars:,}
Cell Types: {adata_ery.obs['CellType'].nunique()}
Donors (Batches): {adata_ery.obs['Donor'].nunique()}

DEVELOPMENTAL TRAJECTORY:
  BFU-E (early progenitor)
    ↓
  CFU-E (committed progenitor)
    ↓
  Pro-Erythroblast
    ↓
  Basophilic Erythroblast
    ↓
  Polychromatic Erythroblast
    ↓
  Orthochromatic Erythroblast
    ↓
  Mature RBC (not in dataset)

MARKER GENES AVAILABLE:
{', '.join(present_markers)}

BATCH CORRECTION GOAL:
- Mix donors within each stage
- Preserve developmental progression
- Validate with marker gene expression
"""
ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
        verticalalignment='center', bbox=dict(boxstyle='round',
        facecolor='wheat', alpha=0.3))

plt.tight_layout()
summary_plot = "erythroid_subset_summary.png"
plt.savefig(summary_plot, dpi=150, bbox_inches='tight')
print(f"   Saved to: {summary_plot}")

print("\n" + "="*80)
print("✓ ERYTHROID SUBSET COMPLETE!")
print("="*80)
print(f"\nOutputs:")
print(f"  1. {output_file} - AnnData with {adata_ery.n_obs:,} erythroid cells")
print(f"  2. {summary_plot} - Summary visualization")
print(f"\nNext steps:")
print(f"  - Run Decipher models on this erythroid-only dataset")
print(f"  - Compare Regular Decipher vs Beta=1.0 (2 heads)")
print(f"  - Validate trajectory using marker genes (HBB, GYPA, etc.)")
print(f"  - Check if V-space shows developmental progression, not donor effects")
