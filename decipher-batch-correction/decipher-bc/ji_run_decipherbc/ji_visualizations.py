"""
Comprehensive visualization script combining batch correction analysis and pseudotime visualization
for adata_combined_2.h5ad

This script creates:
1. Batch correction comparison (3x3 grid) - colored by Delta
2. Biological variable visualizations - time, branch, attention strength
3. V space visualizations colored by pseudotime (latent_t)
4. Side-by-side comparisons of V space: delta vs pseudotime coloring

Original sources:
- ji_visualize_batch_correction.py
- ji_visualize_v_space_by_pseudotime.py
"""

import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import silhouette_score

print("=" * 70)
print("COMPREHENSIVE VISUALIZATION SUITE - adata_combined_2.h5ad")
print("=" * 70)


import os
output_file_path = "ji_run_decipherbc/bc_adata/adata_combined_2_delta_batch_corrected.h5ad"

# Load the results
print("\n[1/5] Loading data...")
adata = sc.read_h5ad(output_file_path)
print(f"✓ Loaded data: {adata.shape[0]} cells, {adata.shape[1]} genes")
print(f"  Available representations:")
for key in adata.obsm.keys():
    print(f"    - {key}: {adata.obsm[key].shape}")

# Check for pseudotime
if 'latent_t' in adata.obs.columns:
    print(f"\n✓ Found pseudotime column: 'latent_t'")
    print(f"  Range: [{adata.obs['latent_t'].min():.3f}, {adata.obs['latent_t'].max():.3f}]")
else:
    print("\n⚠ Warning: 'latent_t' column not found in data!")

# Compute UMAP on batch-corrected embeddings
print("\n[2/5] Computing UMAP on batch-corrected embeddings...")
sc.pp.neighbors(adata, use_rep='X_decipher_batch_corrected_z', key_added='batch_corrected')
sc.tl.umap(adata, neighbors_key='batch_corrected')
adata.obsm['X_decipher_batch_corrected_umap'] = adata.obsm['X_umap'].copy()
print("✓ UMAP computed on batch-corrected embeddings")

# Compute UMAP on original Decipher embeddings for comparison
print("\n[3/5] Computing UMAP on original Decipher embeddings...")
sc.pp.neighbors(adata, use_rep='decipher_decipher_z', key_added='original_decipher')
sc.tl.umap(adata, neighbors_key='original_decipher')
adata.obsm['X_decipher_original_umap'] = adata.obsm['X_umap'].copy()
print("✓ UMAP computed on original Decipher embeddings")

# Define colors for batches
batch_colors = {
    '0': '#1f77b4',
    '0.01': '#ff7f0e',
    '0.05': '#2ca02c',
    '0.1': '#d62728'
}

# Helper function to plot embeddings colored by batch
def plot_embedding(ax, coords, title, adata, color_by='delta'):
    for batch in adata.obs[color_by].unique():
        mask = adata.obs[color_by] == batch
        ax.scatter(coords[mask, 0], coords[mask, 1],
                  c=batch_colors[batch], label=batch, alpha=0.6, s=10)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.legend(title='Delta', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

# ============================================================================
# VISUALIZATION 1: BATCH CORRECTION COMPARISON (3x3 GRID)
# ============================================================================
print("\n[4/5] Creating batch correction comparison visualizations...")

fig = plt.figure(figsize=(20, 12))

# Row 1: UMAP visualizations
ax1 = plt.subplot(2, 4, 1)
plot_embedding(ax1, adata.obsm['X_decipher_original_umap'],
               'Original Decipher (UMAP)', adata)

ax2 = plt.subplot(2, 4, 2)
plot_embedding(ax2, adata.obsm['X_decipher_batch_corrected_umap'],
               'Batch-Corrected Decipher (UMAP)', adata)

ax3 = plt.subplot(2,4, 3)
plot_embedding(ax3, adata.obsm['X_norm_umap'],
               'Default UMAP (Gene Expression)', adata)

# Row 2: Latent space visualizations
ax4 = plt.subplot(2,4, 5)
plot_embedding(ax4, adata.obsm['decipher_decipher_z'][:, :2],
               'Original Decipher Z (2D)', adata)

ax5 = plt.subplot(2,4, 6)
plot_embedding(ax5, adata.obsm['X_decipher_batch_corrected_z'][:, :2],
               'Batch-Corrected Decipher Z (2D)', adata)

#ax6 = plt.subplot(2,4, 6)
#plot_embedding(ax6, adata.obsm['X_pca'][:, :2],
#               'PCA (Gene Expression)', adata)

# Row 3: Component space (v) visualizations
ax7 = plt.subplot(2,4, 7)
plot_embedding(ax7, adata.obsm['decipher_decipher_v'],
               'Original Decipher V (2D)', adata)

ax8 = plt.subplot(2,4, 8)
plot_embedding(ax8, adata.obsm['X_decipher_batch_corrected_v'],
               'Batch-Corrected Decipher V (2D)', adata)

#ax9 = plt.subplot(3, 3, 9)
#plot_embedding(ax9, adata.obsm['X_scVI_umap'],
#               'scVI UMAP', adata)

plt.tight_layout()

out_folder = "ji_visualizations"
os.makedirs(out_folder, exist_ok=True)
suffix = "2"

plt.savefig(
        os.path.join(out_folder, f"batch_correction_comparion_{suffix}.png"),
        dpi=300,
        bbox_inches="tight",
    )
print(f"Saved: batch_correction_comparison_{suffix}.png")
plt.close()


# ============================================================================
# VISUALIZATION 2: BIOLOGICAL VARIABLES
# ============================================================================
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
sc.pl.embedding(adata, basis='decipher_batch_corrected_umap', color='delta',
                show=False, ax=ax6, title='Batch-Corrected - Delta')

plt.tight_layout()
plt.savefig(
        os.path.join(out_folder, f"batch_correction_biological_{suffix}.png"),
        dpi=300,
        bbox_inches="tight",)

print(f"Saved: batch_correction_biological_{suffix}.png")
plt.close()

# ============================================================================
# VISUALIZATION 3: V SPACE COLORED BY PSEUDOTIME
# ============================================================================
print("\n[5/5] Creating V space pseudotime visualizations...")

# Get V space coordinates and pseudotime
v_coords = adata.obsm['decipher_decipher_v']
pseudotime = adata.obs['latent_t'].values

# Flatten pseudotime if it's 2D
if len(pseudotime.shape) > 1:
    pseudotime = pseudotime.flatten()

# Original V space colored by pseudotime
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
scatter = ax.scatter(
    v_coords[:, 0],
    v_coords[:, 1],
    c=pseudotime,
    cmap='viridis',
    alpha=0.6,
    s=10
)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Pseudotime (latent_t)', rotation=270, labelpad=20, fontsize=12)
ax.set_title('Original Decipher V (2D) - Colored by Pseudotime', fontsize=14, fontweight='bold')
ax.set_xlabel('V Dimension 1', fontsize=12)
ax.set_ylabel('V Dimension 2', fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
        os.path.join(out_folder, f"v_space_by_pseudotime_{suffix}.png"),
        dpi=300,
        bbox_inches="tight",)
print("Saved: v_space_by_pseudotime.png")
plt.close()

# Side-by-side comparison: delta vs pseudotime (original V space)
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Left plot: Colored by Delta
ax_left = axes[0]
for batch in adata.obs['delta'].unique():
    mask = adata.obs['delta'] == batch
    ax_left.scatter(
        v_coords[mask, 0],
        v_coords[mask, 1],
        c=batch_colors[batch],
        label=batch,
        alpha=0.6,
        s=10
    )
ax_left.set_title('Original Decipher V (2D) - Colored by Delta',
                   fontsize=14, fontweight='bold')
ax_left.set_xlabel('V Dimension 1', fontsize=12)
ax_left.set_ylabel('V Dimension 2', fontsize=12)
ax_left.legend(title='Delta', bbox_to_anchor=(1.05, 1), loc='upper left')
ax_left.grid(True, alpha=0.3)

# Right plot: Colored by pseudotime
ax_right = axes[1]
scatter = ax_right.scatter(
    v_coords[:, 0],
    v_coords[:, 1],
    c=pseudotime,
    cmap='viridis',
    alpha=0.6,
    s=10
)
cbar = plt.colorbar(scatter, ax=ax_right)
cbar.set_label('Pseudotime (latent_t)', rotation=270, labelpad=20, fontsize=12)
ax_right.set_title('Original Decipher V (2D) - Colored by Pseudotime',
                    fontsize=14, fontweight='bold')
ax_right.set_xlabel('V Dimension 1', fontsize=12)
ax_right.set_ylabel('V Dimension 2', fontsize=12)
ax_right.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
        os.path.join(out_folder, f"v_space_comparison_delta_vs_pseudotime_{suffix}.png"),
        dpi=300,
        bbox_inches="tight",)
print(f"Saved: v_space_comparison_delta_vs_pseudotime_{suffix}.png")
plt.close()

# Batch-Corrected V space visualizations (if available)
if 'X_decipher_batch_corrected_v' in adata.obsm.keys():
    print("\nCreating Batch-Corrected V space visualizations...")
    v_bc_coords = adata.obsm['X_decipher_batch_corrected_v']

    # Single batch-corrected V space plot colored by pseudotime
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    scatter = ax.scatter(
        v_bc_coords[:, 0],
        v_bc_coords[:, 1],
        c=pseudotime,
        cmap='viridis',
        alpha=0.6,
        s=10
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Pseudotime (latent_t)', rotation=270, labelpad=20, fontsize=12)
    ax.set_title('Batch-Corrected Decipher V (2D) - Colored by Pseudotime', fontsize=14, fontweight='bold')
    ax.set_xlabel('V Dimension 1', fontsize=12)
    ax.set_ylabel('V Dimension 2', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_folder, f"v_batch_corrected_by_pseudotime_{suffix}.png"),
        dpi=300,
        bbox_inches="tight",)
    print(f"Saved: v_batch_corrected_by_pseudotime_{suffix}.png")    
    plt.close()

    # Batch-corrected V space side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Left plot: Batch-corrected V space colored by Delta
    ax_left = axes[0]
    for batch in adata.obs['delta'].unique():
        mask = adata.obs['delta'] == batch
        ax_left.scatter(
            v_bc_coords[mask, 0],
            v_bc_coords[mask, 1],
            c=batch_colors[batch],
            label=batch,
            alpha=0.6,
            s=10
        )
    ax_left.set_title('Batch-Corrected Decipher V (2D) - Colored by Delta',
                       fontsize=14, fontweight='bold')
    ax_left.set_xlabel('V Dimension 1', fontsize=12)
    ax_left.set_ylabel('V Dimension 2', fontsize=12)
    ax_left.legend(title='Delta', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_left.grid(True, alpha=0.3)

    # Right plot: Batch-corrected V space colored by pseudotime
    ax_right = axes[1]
    scatter = ax_right.scatter(
        v_bc_coords[:, 0],
        v_bc_coords[:, 1],
        c=pseudotime,
        cmap='viridis',
        alpha=0.6,
        s=10
    )
    cbar = plt.colorbar(scatter, ax=ax_right)
    cbar.set_label('Pseudotime (latent_t)', rotation=270, labelpad=20, fontsize=12)
    ax_right.set_title('Batch-Corrected Decipher V (2D) - Colored by Pseudotime',
                        fontsize=14, fontweight='bold')
    ax_right.set_xlabel('V Dimension 1', fontsize=12)
    ax_right.set_ylabel('V Dimension 2', fontsize=12)
    ax_right.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(out_folder, f"v_batch_corrected_comparison_delta_vs_pseudotime_{suffix}.png"),
        dpi=300,
        bbox_inches="tight",)
    print(f"Saved: v_batch_corrected_comparison_delta_vs_pseudotime_{suffix}.png") 
    plt.close()
else:
    print("\n⚠ Batch-corrected V space (X_decipher_batch_corrected_v) not found in dataset")
    print("  Skipping batch-corrected visualizations.")

# ============================================================================
# BATCH MIXING ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("BATCH MIXING ANALYSIS")
print("=" * 70)

def compute_batch_mixing(embedding, labels):
    """Compute silhouette score (lower = better mixing)"""
    return silhouette_score(embedding, labels)

original_mixing = compute_batch_mixing(
    adata.obsm['decipher_decipher_z'],
    adata.obs['delta']
)
batch_corrected_mixing = compute_batch_mixing(
    adata.obsm['X_decipher_batch_corrected_z'],
    adata.obs['delta']
)

print("\nBatch Mixing Scores (Silhouette, lower = better mixing):")
print(f"  Original Decipher:     {original_mixing:.4f}")
print(f"  Batch-Corrected:       {batch_corrected_mixing:.4f}")
print(f"  Improvement:           {original_mixing - batch_corrected_mixing:.4f}")
if batch_corrected_mixing < original_mixing:
    print("  ✓ Batch-corrected shows BETTER mixing (lower score)")
else:
    print("  ⚠ Original shows better mixing")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("\nFiles created:")
print("\n1. batch_correction_comparison_2.png")
print("   - 3x3 grid comparing different methods")
print("   - All colored by Delta")
print("   - Shows UMAP, latent space (z), and component space (v)")

print("\n2. batch_correction_biological_2.png")
print("   - Biological variables (latent_t, branch_id)")
print("   - Attention strength visualization")
print("   - Delta comparison")

print("\n3. v_space_by_pseudotime.png")
print("   - Original Decipher V space colored by pseudotime")

print("\n4. v_space_comparison_delta_vs_pseudotime.png")
print("   - Side-by-side comparison:")
print("     Left:  Original V space colored by Delta")
print("     Right: Original V space colored by pseudotime (latent_t)")

if 'X_decipher_batch_corrected_v' in adata.obsm.keys():
    print("\n5. v_batch_corrected_by_pseudotime.png")
    print("   - Batch-Corrected Decipher V space colored by pseudotime")

    print("\n6. v_batch_corrected_comparison_delta_vs_pseudotime.png")
    print("   - Side-by-side comparison:")
    print("     Left:  Batch-corrected V space colored by Delta")
    print("     Right: Batch-corrected V space colored by pseudotime (latent_t)")

print("\nVisualization guide:")
print("  - Look for MIXING of colors (batches) in batch-corrected plots")
print("  - Better batch correction = more mixed colors")
print("  - Biological structure should be PRESERVED")
print("  - Check that time (latent_t) and branches are still visible")
print("  - Pseudotime shows temporal progression of cells")
print("  - Compare delta vs pseudotime plots to see batch effects vs biological time")

print("\nQuantitative metrics:")
print(f"  Batch mixing (Silhouette):")
print(f"    Original:        {original_mixing:.4f}")
print(f"    Batch-corrected: {batch_corrected_mixing:.4f}")
print(f"    Improvement:     {original_mixing - batch_corrected_mixing:.4f}")

print("\n" + "=" * 70)
print("✓ COMPREHENSIVE VISUALIZATION COMPLETE!")
print("=" * 70)
print("\nOpen the PNG files to view the results!")
