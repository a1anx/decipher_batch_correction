"""
Visualization to demonstrate the benefit of batch correction
by comparing how batch effects (alpha-shift) are mixed before vs after correction
"""

import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np

print("=" * 70)
print("DEMONSTRATING BATCH CORRECTION BENEFIT")
print("=" * 70)

# Load data
print("\nLoading data...")
adata = sc.read_h5ad('../../simulation/adata/adata_combined_2_batch_corrected.h5ad')
print(f"✓ Loaded: {adata.shape[0]} cells, {adata.shape[1]} genes")

# Define colors for batches
batch_colors = {
    'False': '#1f77b4',
    'Alpha 0.01': '#ff7f0e',
    'Alpha 0.05': '#2ca02c',
    'Alpha 0.1': '#d62728'
}

# Get data
v_orig = adata.obsm['decipher_decipher_v']
v_bc = adata.obsm['X_decipher_batch_corrected_v']
pseudotime = adata.obs['latent_t'].values
if len(pseudotime.shape) > 1:
    pseudotime = pseudotime.flatten()

print("\nCreating comprehensive comparison figure...")

# Create 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# Row 1: Original Decipher V
# Left: colored by batch (shows batch effects)
ax = axes[0, 0]
for batch in adata.obs['shift'].unique():
    mask = adata.obs['shift'] == batch
    ax.scatter(v_orig[mask, 0], v_orig[mask, 1],
               c=batch_colors[batch], label=batch, alpha=0.6, s=15)
ax.set_title('Original Decipher V - Colored by Batch\n(Shows BATCH EFFECTS - clusters by color = BAD)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('V Dimension 1', fontsize=12)
ax.set_ylabel('V Dimension 2', fontsize=12)
ax.legend(title='Batch', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3)

# Right: colored by pseudotime (shows biology)
ax = axes[0, 1]
scatter = ax.scatter(v_orig[:, 0], v_orig[:, 1],
                     c=pseudotime, cmap='viridis', alpha=0.6, s=15)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Pseudotime', rotation=270, labelpad=20, fontsize=12)
ax.set_title('Original Decipher V - Colored by Pseudotime\n(Shows BIOLOGY preserved)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('V Dimension 1', fontsize=12)
ax.set_ylabel('V Dimension 2', fontsize=12)
ax.grid(True, alpha=0.3)

# Row 2: Batch-Corrected Decipher V
# Left: colored by batch (should be mixed)
ax = axes[1, 0]
for batch in adata.obs['shift'].unique():
    mask = adata.obs['shift'] == batch
    ax.scatter(v_bc[mask, 0], v_bc[mask, 1],
               c=batch_colors[batch], label=batch, alpha=0.6, s=15)
ax.set_title('Batch-Corrected V - Colored by Batch\n(Batches MIXED = GOOD correction)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('V Dimension 1', fontsize=12)
ax.set_ylabel('V Dimension 2', fontsize=12)
ax.legend(title='Batch', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3)

# Right: colored by pseudotime (should preserve biology)
ax = axes[1, 1]
scatter = ax.scatter(v_bc[:, 0], v_bc[:, 1],
                     c=pseudotime, cmap='viridis', alpha=0.6, s=15)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Pseudotime', rotation=270, labelpad=20, fontsize=12)
ax.set_title('Batch-Corrected V - Colored by Pseudotime\n(BIOLOGY still preserved)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('V Dimension 1', fontsize=12)
ax.set_ylabel('V Dimension 2', fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_file = 'batch_correction_benefit_demonstration.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_file}")
plt.close()

# Compute quantitative metrics
print("\n" + "=" * 70)
print("QUANTITATIVE ASSESSMENT")
print("=" * 70)

from sklearn.metrics import silhouette_score

# Batch mixing score (lower = better mixing = better batch correction)
orig_batch_mixing = silhouette_score(v_orig, adata.obs['shift'])
bc_batch_mixing = silhouette_score(v_bc, adata.obs['shift'])

print("\nBatch Mixing (Silhouette Score):")
print("  Lower = better mixing of batches = better correction")
print(f"  Original Decipher V:      {orig_batch_mixing:.4f}")
print(f"  Batch-Corrected V:        {bc_batch_mixing:.4f}")
print(f"  Improvement:              {orig_batch_mixing - bc_batch_mixing:.4f}")

if bc_batch_mixing < orig_batch_mixing:
    improvement_pct = ((orig_batch_mixing - bc_batch_mixing) / abs(orig_batch_mixing)) * 100
    print(f"  → {improvement_pct:.1f}% improvement in batch mixing! ✓")
else:
    print("  → No improvement in batch mixing ✗")

# Biology preservation (higher = better preservation of pseudotime structure)
from scipy.stats import spearmanr

# Check if pseudotime correlates with first PC of embeddings
from sklearn.decomposition import PCA
pca_orig = PCA(n_components=1).fit_transform(v_orig).flatten()
pca_bc = PCA(n_components=1).fit_transform(v_bc).flatten()

corr_orig, _ = spearmanr(pseudotime, pca_orig)
corr_bc, _ = spearmanr(pseudotime, pca_bc)

print("\nBiology Preservation (Pseudotime-PC1 Correlation):")
print("  Higher = better preservation of biological trajectory")
print(f"  Original Decipher V:      {abs(corr_orig):.4f}")
print(f"  Batch-Corrected V:        {abs(corr_bc):.4f}")
print(f"  Change:                   {abs(corr_bc) - abs(corr_orig):.4f}")

if abs(corr_bc) >= abs(corr_orig) * 0.9:  # Allow 10% tolerance
    print(f"  → Biology well preserved! ✓")
else:
    print(f"  → Biology may be compromised ⚠")

print("\n" + "=" * 70)
print("INTERPRETATION GUIDE")
print("=" * 70)
print("\nWhat to look for in the plots:")
print("  TOP ROW (Original):")
print("    - Left: If colors cluster together = batch effects present")
print("    - Right: If pseudotime shows gradient = biology captured")
print("\n  BOTTOM ROW (Batch-Corrected):")
print("    - Left: If colors are MIXED throughout = batch correction worked!")
print("    - Right: If pseudotime gradient preserved = biology maintained!")
print("\nGood batch correction should:")
print("  1. MIX the batch colors (left column: top→bottom)")
print("  2. PRESERVE the pseudotime gradient (right column: top≈bottom)")

print("\n" + "=" * 70)
print("✓ ANALYSIS COMPLETE!")
print("=" * 70)
