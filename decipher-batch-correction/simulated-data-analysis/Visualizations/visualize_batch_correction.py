"""
Visualization script to compare original Decipher vs Batch-Corrected Decipher

This script creates side-by-side comparisons of:
1. Original Decipher embeddings
2. Batch-corrected Decipher embeddings
3. Other dimensionality reduction methods (PCA, UMAP, etc.)

All colored by batch (shift) to assess batch correction quality
"""

import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

print("=" * 70)
print("VISUALIZING BATCH CORRECTION RESULTS")
print("=" * 70)

# Load the results
print("\n[1/4] Loading data...")
adata = sc.read_h5ad('simulation/adata/adata_combined_1_batch_corrected.h5ad')
print(f"✓ Loaded data: {adata.shape[0]} cells, {adata.shape[1]} genes")
print(f"  Available representations:")
for key in adata.obsm.keys():
    print(f"    - {key}: {adata.obsm[key].shape}")

# Compute UMAP on batch-corrected embeddings
print("\n[2/4] Computing UMAP on batch-corrected embeddings...")
sc.pp.neighbors(adata, use_rep='X_decipher_batch_corrected_z', key_added='batch_corrected')
sc.tl.umap(adata, neighbors_key='batch_corrected')
adata.obsm['X_decipher_batch_corrected_umap'] = adata.obsm['X_umap'].copy()
print("✓ UMAP computed on batch-corrected embeddings")

# Compute UMAP on original Decipher embeddings for comparison
print("\n[3/4] Computing UMAP on original Decipher embeddings...")
sc.pp.neighbors(adata, use_rep='decipher_decipher_z', key_added='original_decipher')
sc.tl.umap(adata, neighbors_key='original_decipher')
adata.obsm['X_decipher_original_umap'] = adata.obsm['X_umap'].copy()
print("✓ UMAP computed on original Decipher embeddings")

# Create comprehensive visualization
print("\n[4/4] Creating visualizations...")

# Set up the figure with multiple subplots
fig = plt.figure(figsize=(20, 12))

# Define colors for batches
batch_colors = {
    'False': '#1f77b4',
    'Alpha 0.001': '#ff7f0e',
    'Alpha 0.005': '#2ca02c',
    'Alpha 0.01': '#d62728'
}

# Helper function to plot embeddings
def plot_embedding(ax, coords, title, adata, color_by='shift'):
    for batch in adata.obs[color_by].unique():
        mask = adata.obs[color_by] == batch
        ax.scatter(coords[mask, 0], coords[mask, 1],
                  c=batch_colors[batch], label=batch, alpha=0.6, s=10)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

# Row 1: UMAP visualizations
ax1 = plt.subplot(3, 3, 1)
plot_embedding(ax1, adata.obsm['X_decipher_original_umap'],
               'Original Decipher (UMAP)', adata)

ax2 = plt.subplot(3, 3, 2)
plot_embedding(ax2, adata.obsm['X_decipher_batch_corrected_umap'],
               'Batch-Corrected Decipher (UMAP)', adata)

ax3 = plt.subplot(3, 3, 3)
plot_embedding(ax3, adata.obsm['X_default_umap'],
               'Default UMAP (Gene Expression)', adata)

# Row 2: Latent space visualizations
ax4 = plt.subplot(3, 3, 4)
# For original Decipher, plot first 2 dimensions of z
plot_embedding(ax4, adata.obsm['decipher_decipher_z'][:, :2],
               'Original Decipher Z (2D)', adata)

ax5 = plt.subplot(3, 3, 5)
# For batch-corrected, plot first 2 dimensions of z
plot_embedding(ax5, adata.obsm['X_decipher_batch_corrected_z'][:, :2],
               'Batch-Corrected Decipher Z (2D)', adata)

ax6 = plt.subplot(3, 3, 6)
# Plot PCA
plot_embedding(ax6, adata.obsm['X_pca'][:, :2],
               'PCA (Gene Expression)', adata)

# Row 3: Component space (v) visualizations
ax7 = plt.subplot(3, 3, 7)
plot_embedding(ax7, adata.obsm['decipher_decipher_v'],
               'Original Decipher V (2D)', adata)

ax8 = plt.subplot(3, 3, 8)
plot_embedding(ax8, adata.obsm['X_decipher_batch_corrected_v'],
               'Batch-Corrected Decipher V (2D)', adata)

ax9 = plt.subplot(3, 3, 9)
plot_embedding(ax9, adata.obsm['X_scVI_umap'],
               'scVI UMAP', adata)

plt.tight_layout()
plt.savefig('batch_correction_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: batch_correction_comparison.png")
plt.close()

# Create additional plot: Biological variables
print("\nCreating biological variable visualizations...")
fig2 = plt.figure(figsize=(20, 8))

# Plot by latent_t (time)
ax1 = plt.subplot(2, 3, 1)
sc.pl.embedding(adata, basis='decipher_original_umap', color='latent_t',
                show=False, ax=ax1, title='Original Decipher - Time')

ax2 = plt.subplot(2, 3, 2)
sc.pl.embedding(adata, basis='decipher_batch_corrected_umap', color='latent_t',
                show=False, ax=ax2, title='Batch-Corrected - Time')

ax3 = plt.subplot(2, 3, 3)
sc.pl.embedding(adata, basis='decipher_batch_corrected_umap', color='batch_attention_strength',
                show=False, ax=ax3, title='Attention Strength')

# Plot by branch_id
ax4 = plt.subplot(2, 3, 4)
sc.pl.embedding(adata, basis='decipher_original_umap', color='branch_id',
                show=False, ax=ax4, title='Original Decipher - Branch')

ax5 = plt.subplot(2, 3, 5)
sc.pl.embedding(adata, basis='decipher_batch_corrected_umap', color='branch_id',
                show=False, ax=ax5, title='Batch-Corrected - Branch')

ax6 = plt.subplot(2, 3, 6)
sc.pl.embedding(adata, basis='decipher_batch_corrected_umap', color='shift',
                show=False, ax=ax6, title='Batch-Corrected - Condition')

plt.tight_layout()
plt.savefig('batch_correction_biological.png', dpi=300, bbox_inches='tight')
print("✓ Saved: batch_correction_biological.png")
plt.close()

# Create batch mixing analysis
print("\nAnalyzing batch mixing...")
from sklearn.metrics import silhouette_score

def compute_batch_mixing(embedding, labels):
    """Compute silhouette score (lower = better mixing)"""
    return silhouette_score(embedding, labels)

original_mixing = compute_batch_mixing(
    adata.obsm['decipher_decipher_z'],
    adata.obs['shift']
)
batch_corrected_mixing = compute_batch_mixing(
    adata.obsm['X_decipher_batch_corrected_z'],
    adata.obs['shift']
)

print("\nBatch Mixing Scores (Silhouette, lower = better mixing):")
print(f"  Original Decipher:     {original_mixing:.4f}")
print(f"  Batch-Corrected:       {batch_corrected_mixing:.4f}")
if batch_corrected_mixing < original_mixing:
    print("  ✓ Batch-corrected shows BETTER mixing (lower score)")
else:
    print("  ⚠ Original shows better mixing")

# Create loss curve plot
print("\nNote: Loss curves are not saved in the AnnData object.")
print("      To visualize loss curves, modify train_batch_corrected.py to save them.")

# Summary statistics
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("\nFiles created:")
print("  1. batch_correction_comparison.png")
print("     - 3x3 grid comparing different methods")
print("     - All colored by condition (shift)")
print("     - Shows UMAP, latent space (z), and component space (v)")

print("\n  2. batch_correction_biological.png")
print("     - Biological variables (latent_t, branch_id)")
print("     - Attention strength visualization")
print("     - Condition comparison")

print("\nVisualization guide:")
print("  - Look for MIXING of colors (batches) in batch-corrected plots")
print("  - Better batch correction = more mixed colors")
print("  - Biological structure should be PRESERVED")
print("  - Check that time (latent_t) and branches are still visible")

print("\nQuantitative metrics:")
print(f"  Batch mixing (Silhouette):")
print(f"    Original:        {original_mixing:.4f}")
print(f"    Batch-corrected: {batch_corrected_mixing:.4f}")
print(f"    Improvement:     {original_mixing - batch_corrected_mixing:.4f}")

print("\n" + "=" * 70)
print("✓ VISUALIZATION COMPLETE!")
print("=" * 70)
print("\nOpen the PNG files to view the results!")
