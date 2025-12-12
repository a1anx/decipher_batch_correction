"""
Comprehensive visualization script for concatenation-based batch correction on delta shift data

This script combines:
1. General batch correction comparison
2. V space by pseudotime

Creates all visualizations in one run for delta shift analysis.
"""

import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import silhouette_score

print("=" * 80)
print("COMPREHENSIVE VISUALIZATION: CONCATENATION-BASED BATCH CORRECTION (DELTA)")
print("=" * 80)

# ==============================================================================
# PART 1: LOAD DATA AND COMPUTE UMAPS
# ==============================================================================

print("\n[1/5] Loading data...")
adata = sc.read_h5ad('/home/alan/Documents/decipher_batch_correction/simulation/archive/oldadata_alpha/adata_combined_2_delta_concat.h5ad')
print(f"✓ Loaded data: {adata.shape[0]} cells, {adata.shape[1]} genes")
print(f"  Available representations:")
for key in adata.obsm.keys():
    print(f"    - {key}: {adata.obsm[key].shape}")

# Check for pseudotime
if 'latent_t' in adata.obs.columns:
    print(f"\n✓ Found pseudotime column: 'latent_t'")
    print(f"  Range: [{adata.obs['latent_t'].min():.3f}, {adata.obs['latent_t'].max():.3f}]")
    pseudotime = adata.obs['latent_t'].values
    if len(pseudotime.shape) > 1:
        pseudotime = pseudotime.flatten()
else:
    print("\n⚠ Warning: 'latent_t' column not found!")
    pseudotime = None

print("\n[2/5] Computing UMAPs...")
# UMAP on concatenation embeddings
sc.pp.neighbors(adata, use_rep='X_decipher_concat_z', key_added='concat_corrected')
sc.tl.umap(adata, neighbors_key='concat_corrected')
adata.obsm['X_decipher_concat_umap'] = adata.obsm['X_umap'].copy()
print("✓ UMAP on concatenation embeddings")

# UMAP on original Decipher
sc.pp.neighbors(adata, use_rep='decipher_decipher_z', key_added='original_decipher')
sc.tl.umap(adata, neighbors_key='original_decipher')
adata.obsm['X_decipher_original_umap'] = adata.obsm['X_umap'].copy()
print("✓ UMAP on original Decipher embeddings")

# UMAP on normalized gene expression
sc.pp.neighbors(adata, use_rep='X_norm_umap', key_added='default')
sc.tl.umap(adata, neighbors_key='default')
adata.obsm['X_default_umap'] = adata.obsm['X_umap'].copy()
print("✓ UMAP on normalized gene expression")

# ==============================================================================
# PART 2: COMPUTE METRICS
# ==============================================================================

print("\n[3/5] Computing batch mixing metrics...")

def compute_batch_mixing(embedding, labels):
    """Compute silhouette score (lower = better mixing)"""
    return silhouette_score(embedding, labels)

original_mixing = compute_batch_mixing(
    adata.obsm['decipher_decipher_z'],
    adata.obs['delta']
)
concat_mixing = compute_batch_mixing(
    adata.obsm['X_decipher_concat_z'],
    adata.obs['delta']
)

print(f"\nBatch Mixing Scores (Silhouette, lower = better mixing):")
print(f"  Original Decipher:      {original_mixing:.4f}")
print(f"  Concat Batch-Corrected: {concat_mixing:.4f}")
print(f"  Improvement:            {original_mixing - concat_mixing:+.4f}")
if concat_mixing < original_mixing:
    print("  ✓ Concatenation shows BETTER mixing")
else:
    print("  ⚠ Original shows better mixing")

# ==============================================================================
# PART 3: CREATE VISUALIZATIONS
# ==============================================================================

print("\n[4/5] Creating visualizations...")

# Define colors for delta values (use discrete colormap)
delta_values = sorted(adata.obs['delta'].unique())
cmap = plt.colormaps.get_cmap('tab10')
batch_colors = {val: cmap(i) for i, val in enumerate(delta_values)}

def plot_embedding(ax, coords, title, adata, color_by='delta'):
    """Helper function to plot embeddings colored by batches"""
    for batch in sorted(adata.obs[color_by].unique()):
        mask = adata.obs[color_by] == batch
        ax.scatter(coords[mask, 0], coords[mask, 1],
                  c=[batch_colors[batch]], label=f'Delta={batch}', alpha=0.6, s=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.legend(title='Delta Value', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

# ==============================================================================
# FIGURE 1: 3×3 Method Comparison Grid
# ==============================================================================
print("  Creating Figure 1: Method comparison (3×3 grid)...")
fig = plt.figure(figsize=(20, 12))

# Row 1: UMAP visualizations
ax1 = plt.subplot(3, 3, 1)
plot_embedding(ax1, adata.obsm['X_decipher_original_umap'],
               'Original Decipher (UMAP)', adata)

ax2 = plt.subplot(3, 3, 2)
plot_embedding(ax2, adata.obsm['X_decipher_concat_umap'],
               'Concat Batch-Corrected (UMAP)', adata)

ax3 = plt.subplot(3, 3, 3)
plot_embedding(ax3, adata.obsm['X_default_umap'],
               'Default UMAP (Gene Expression)', adata)

# Row 2: Latent space (Z) visualizations
ax4 = plt.subplot(3, 3, 4)
plot_embedding(ax4, adata.obsm['decipher_decipher_z'][:, :2],
               'Original Decipher Z (2D)', adata)

ax5 = plt.subplot(3, 3, 5)
plot_embedding(ax5, adata.obsm['X_decipher_concat_z'][:, :2],
               'Concat Batch-Corrected Z (2D)', adata)

ax6 = plt.subplot(3, 3, 6)
# Use latent_z for PCA alternative
if 'latent_z' in adata.obsm:
    plot_embedding(ax6, adata.obsm['latent_z'][:, :2],
                   'True Latent Z (Ground Truth)', adata)
else:
    ax6.text(0.5, 0.5, 'No PCA/Latent Available',
             ha='center', va='center', transform=ax6.transAxes)
    ax6.set_title('Ground Truth Latent', fontsize=12, fontweight='bold')

# Row 3: Component space (V) visualizations
ax7 = plt.subplot(3, 3, 7)
plot_embedding(ax7, adata.obsm['decipher_decipher_v'],
               'Original Decipher V (2D)', adata)

ax8 = plt.subplot(3, 3, 8)
plot_embedding(ax8, adata.obsm['X_decipher_concat_v'],
               'Concat Batch-Corrected V (2D)', adata)

ax9 = plt.subplot(3, 3, 9)
# Empty panel or additional visualization
ax9.text(0.5, 0.5, 'Reserved for\nAdditional Analysis',
         ha='center', va='center', transform=ax9.transAxes)
ax9.set_title('Reserved', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('concat_method_comparison_delta.png', dpi=300, bbox_inches='tight')
print("    ✓ Saved: concat_method_comparison_delta.png")
plt.close()

# ==============================================================================
# FIGURE 2: Biological Variables
# ==============================================================================
print("  Creating Figure 2: Biological variables...")
fig2 = plt.figure(figsize=(20, 8))

# Plot by latent_t (time)
ax1 = plt.subplot(2, 3, 1)
sc.pl.embedding(adata, basis='decipher_original_umap', color='latent_t',
                show=False, ax=ax1, title='Original Decipher - Time')

ax2 = plt.subplot(2, 3, 2)
sc.pl.embedding(adata, basis='decipher_concat_umap', color='latent_t',
                show=False, ax=ax2, title='Concat Batch-Corrected - Time')

# Variance explained
ax3 = plt.subplot(2, 3, 3)
z_var = np.var(adata.obsm['X_decipher_concat_z'], axis=0)
z_var_pct = 100 * z_var / z_var.sum()
ax3.bar(range(1, len(z_var_pct) + 1), z_var_pct)
ax3.set_xlabel('Latent Dimension')
ax3.set_ylabel('Variance Explained (%)')
ax3.set_title('Concat: Variance per Dimension')
ax3.grid(True, alpha=0.3)

# Plot by branch_id
ax4 = plt.subplot(2, 3, 4)
sc.pl.embedding(adata, basis='decipher_original_umap', color='branch_id',
                show=False, ax=ax4, title='Original Decipher - Branch')

ax5 = plt.subplot(2, 3, 5)
sc.pl.embedding(adata, basis='decipher_concat_umap', color='branch_id',
                show=False, ax=ax5, title='Concat Batch-Corrected - Branch')

ax6 = plt.subplot(2, 3, 6)
sc.pl.embedding(adata, basis='decipher_concat_umap', color='delta',
                show=False, ax=ax6, title='Concat Batch-Corrected - Delta')

plt.tight_layout()
plt.savefig('concat_biological_variables_delta.png', dpi=300, bbox_inches='tight')
print("    ✓ Saved: concat_biological_variables_delta.png")
plt.close()

# ==============================================================================
# FIGURE 3: V Space - Four Panel Comparison
# ==============================================================================
if pseudotime is not None:
    print("  Creating Figure 3: V space four-panel comparison...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    v_original = adata.obsm['decipher_decipher_v']
    v_concat = adata.obsm['X_decipher_concat_v']

    # Panel 1: Original V by delta
    ax = axes[0, 0]
    for batch in sorted(adata.obs['delta'].unique()):
        mask = adata.obs['delta'] == batch
        ax.scatter(v_original[mask, 0], v_original[mask, 1],
                  c=[batch_colors[batch]], label=f'Delta={batch}', alpha=0.6, s=10)
    ax.set_title('Original V - by Delta', fontsize=12, fontweight='bold')
    ax.set_xlabel('V Dimension 1')
    ax.set_ylabel('V Dimension 2')
    ax.legend(title='Delta Value', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Original V by pseudotime
    ax = axes[0, 1]
    scatter = ax.scatter(v_original[:, 0], v_original[:, 1],
                        c=pseudotime, cmap='viridis', alpha=0.6, s=10)
    plt.colorbar(scatter, ax=ax, label='Pseudotime')
    ax.set_title('Original V - by Pseudotime', fontsize=12, fontweight='bold')
    ax.set_xlabel('V Dimension 1')
    ax.set_ylabel('V Dimension 2')
    ax.grid(True, alpha=0.3)

    # Panel 3: Concat V by delta
    ax = axes[1, 0]
    for batch in sorted(adata.obs['delta'].unique()):
        mask = adata.obs['delta'] == batch
        ax.scatter(v_concat[mask, 0], v_concat[mask, 1],
                  c=[batch_colors[batch]], label=f'Delta={batch}', alpha=0.6, s=10)
    ax.set_title('Concat V - by Delta', fontsize=12, fontweight='bold')
    ax.set_xlabel('V Dimension 1')
    ax.set_ylabel('V Dimension 2')
    ax.legend(title='Delta Value', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Concat V by pseudotime
    ax = axes[1, 1]
    scatter = ax.scatter(v_concat[:, 0], v_concat[:, 1],
                        c=pseudotime, cmap='viridis', alpha=0.6, s=10)
    plt.colorbar(scatter, ax=ax, label='Pseudotime')
    ax.set_title('Concat V - by Pseudotime', fontsize=12, fontweight='bold')
    ax.set_xlabel('V Dimension 1')
    ax.set_ylabel('V Dimension 2')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('concat_v_space_four_panel_delta.png', dpi=300, bbox_inches='tight')
    print("    ✓ Saved: concat_v_space_four_panel_delta.png")
    plt.close()

# ==============================================================================
# FIGURE 4: V Space - Side-by-Side Comparisons
# ==============================================================================
if pseudotime is not None:
    print("  Creating Figure 4: V space side-by-side comparisons...")

    # Original V space
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax_left = axes[0]
    for batch in sorted(adata.obs['delta'].unique()):
        mask = adata.obs['delta'] == batch
        ax_left.scatter(v_original[mask, 0], v_original[mask, 1],
                       c=[batch_colors[batch]], label=f'Delta={batch}', alpha=0.6, s=10)
    ax_left.set_title('Original V - by Delta', fontsize=14, fontweight='bold')
    ax_left.set_xlabel('V Dimension 1', fontsize=12)
    ax_left.set_ylabel('V Dimension 2', fontsize=12)
    ax_left.legend(title='Delta Value', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_left.grid(True, alpha=0.3)

    ax_right = axes[1]
    scatter = ax_right.scatter(v_original[:, 0], v_original[:, 1],
                              c=pseudotime, cmap='viridis', alpha=0.6, s=10)
    cbar = plt.colorbar(scatter, ax=ax_right)
    cbar.set_label('Pseudotime (latent_t)', rotation=270, labelpad=20, fontsize=12)
    ax_right.set_title('Original V - by Pseudotime', fontsize=14, fontweight='bold')
    ax_right.set_xlabel('V Dimension 1', fontsize=12)
    ax_right.set_ylabel('V Dimension 2', fontsize=12)
    ax_right.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('concat_v_original_comparison_delta.png', dpi=300, bbox_inches='tight')
    print("    ✓ Saved: concat_v_original_comparison_delta.png")
    plt.close()

    # Concat V space
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax_left = axes[0]
    for batch in sorted(adata.obs['delta'].unique()):
        mask = adata.obs['delta'] == batch
        ax_left.scatter(v_concat[mask, 0], v_concat[mask, 1],
                       c=[batch_colors[batch]], label=f'Delta={batch}', alpha=0.6, s=10)
    ax_left.set_title('Concat V - by Delta', fontsize=14, fontweight='bold')
    ax_left.set_xlabel('V Dimension 1', fontsize=12)
    ax_left.set_ylabel('V Dimension 2', fontsize=12)
    ax_left.legend(title='Delta Value', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_left.grid(True, alpha=0.3)

    ax_right = axes[1]
    scatter = ax_right.scatter(v_concat[:, 0], v_concat[:, 1],
                              c=pseudotime, cmap='viridis', alpha=0.6, s=10)
    cbar = plt.colorbar(scatter, ax=ax_right)
    cbar.set_label('Pseudotime (latent_t)', rotation=270, labelpad=20, fontsize=12)
    ax_right.set_title('Concat V - by Pseudotime', fontsize=14, fontweight='bold')
    ax_right.set_xlabel('V Dimension 1', fontsize=12)
    ax_right.set_ylabel('V Dimension 2', fontsize=12)
    ax_right.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('concat_v_batch_corrected_comparison_delta.png', dpi=300, bbox_inches='tight')
    print("    ✓ Saved: concat_v_batch_corrected_comparison_delta.png")
    plt.close()

# ==============================================================================
# PART 5: SUMMARY REPORT
# ==============================================================================

print("\n[5/5] Generating summary report...")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("\nFiles created:")
print("  1. concat_method_comparison_delta.png")
print("     - 3×3 grid comparing different methods")
print("     - All colored by delta value")
print("     - UMAP, Z space, V space across methods")

print("\n  2. concat_biological_variables_delta.png")
print("     - Biological variables (latent_t, branch_id)")
print("     - Variance explained visualization")
print("     - Delta value comparison")

if pseudotime is not None:
    print("\n  3. concat_v_space_four_panel_delta.png")
    print("     - Four-panel V space comparison")
    print("     - Top: Original V (delta vs pseudotime)")
    print("     - Bottom: Concat V (delta vs pseudotime)")

    print("\n  4. concat_v_original_comparison_delta.png")
    print("     - Original V: delta vs pseudotime side-by-side")

    print("\n  5. concat_v_batch_corrected_comparison_delta.png")
    print("     - Concat V: delta vs pseudotime side-by-side")

print("\nApproach:")
print("  - Uses CONCATENATION: concat([z, batch_emb]) → MLP")
print("  - Simpler than attention mechanism")
print("  - Faster training")
print("  - Good baseline for comparison")

print("\nBatch Mixing Performance:")
print(f"  Original:           {original_mixing:.4f}")
print(f"  Concat-corrected:   {concat_mixing:.4f}")
print(f"  Improvement:        {original_mixing - concat_mixing:+.4f}")

if concat_mixing < original_mixing:
    print("  ✓ Batch correction successful (lower silhouette score)")
else:
    print("  ⚠ Original performs better")

print("\nInterpretation guide:")
print("  - Look for MIXING of colors (delta values) in concat plots")
print("  - Better batch correction = more mixed colors")
print("  - Biological structure should be PRESERVED")
print("  - Check that time (latent_t) and branches are still visible")
print("  - Smooth pseudotime gradient = trajectory preserved")

print("\n" + "=" * 80)
print("✓ ALL VISUALIZATIONS COMPLETE!")
print("=" * 80)
print("\nNext step: Compare with attention using compare_concat_vs_attention_delta.py")
