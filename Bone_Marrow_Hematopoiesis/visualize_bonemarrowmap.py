"""
Comprehensive visualization script for BoneMarrowMap batch correction results

This script combines:
- visualize_batch_correction_2.py (batch mixing analysis)
- visualize_v_space_by_pseudotime.py (trajectory visualization)

Creates:
1. Batch correction quality plots (UMAP, mixing scores)
2. Biological signal preservation (cell types, trajectories)
3. Attention analysis (heatmaps, distributions)
"""

import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
import os
from sklearn.metrics import silhouette_score

print("=" * 80)
print("BONEMARROWMAP - Visualization of Batch Correction Results")
print("=" * 80)

# Load results
print("\n[1/4] Loading batch-corrected data...")
input_file = "bonemarrowmap_batch_corrected.h5ad"

if not os.path.exists(input_file):
    print(f"\n❌ ERROR: Results file not found: {input_file}")
    print(f"   Please run: python run_bonemarrowmap_analysis.py")
    sys.exit(1)

adata = sc.read_h5ad(input_file)
print(f"✓ Loaded data: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")

print(f"\nAvailable representations:")
for key in sorted(adata.obsm.keys()):
    print(f"  - {key}: {adata.obsm[key].shape}")

# Auto-detect batch and cell type columns
print("\n[2/4] Detecting metadata columns...")

# Batch/donor column
donor_cols = [col for col in adata.obs.columns
              if 'donor' in col.lower() or 'batch' in col.lower()
              or 'sample' in col.lower() or 'patient' in col.lower()]
batch_key = donor_cols[0] if donor_cols else None

if not batch_key:
    print("❌ ERROR: No batch column found!")
    sys.exit(1)

print(f"✓ Batch column: '{batch_key}' ({adata.obs[batch_key].nunique()} batches)")

# Cell type column
cell_type_cols = [col for col in adata.obs.columns
                  if 'type' in col.lower() or 'cluster' in col.lower()
                  or 'celltype' in col.lower() or 'annotation' in col.lower()]
cell_type_key = cell_type_cols[0] if cell_type_cols else None

if cell_type_key:
    print(f"✓ Cell type column: '{cell_type_key}' ({adata.obs[cell_type_key].nunique()} types)")
else:
    print("⚠ No cell type column detected")

# Pseudotime column (optional)
pseudotime_cols = [col for col in adata.obs.columns
                   if 'time' in col.lower() or 'pseudotime' in col.lower()
                   or 'dpt' in col.lower()]
pseudotime_key = pseudotime_cols[0] if pseudotime_cols else None

if pseudotime_key:
    print(f"✓ Pseudotime column: '{pseudotime_key}'")
else:
    print("⚠ No pseudotime column detected (optional)")

# Compute metrics
print("\n[3/4] Computing batch mixing metrics...")

# Silhouette score for batch mixing (lower = better mixing)
if 'X_decipher_batch_corrected_z' in adata.obsm:
    batch_corrected_silhouette = silhouette_score(
        adata.obsm['X_decipher_batch_corrected_z'],
        adata.obs[batch_key]
    )
    print(f"  Batch-corrected silhouette score: {batch_corrected_silhouette:.4f}")
else:
    batch_corrected_silhouette = None

# Create visualizations
print("\n[4/4] Creating visualizations...")

# =====================================================================
# Figure 1: Batch Correction Comparison (3x3 grid)
# =====================================================================
print("\n  Creating batch correction comparison plot...")

fig = plt.figure(figsize=(20, 16))

# Define color palette for batches (use first 20 batches for clarity)
n_batches = adata.obs[batch_key].nunique()
if n_batches <= 20:
    batch_palette = sns.color_palette('tab20', n_batches)
else:
    batch_palette = sns.color_palette('tab20', 20)
    print(f"    Note: Using colors for first 20 of {n_batches} batches")

# Helper function for scatter plots
def plot_scatter(ax, coords, color_by, title, palette=None, legend=True):
    """Plot scatter with categorical or continuous coloring"""
    if color_by in adata.obs.columns:
        values = adata.obs[color_by]

        # Check if categorical or continuous
        if values.dtype == 'object' or values.dtype.name == 'category':
            # Categorical
            unique_vals = values.unique()
            if palette is None:
                palette = sns.color_palette('tab10', len(unique_vals))

            for i, val in enumerate(unique_vals[:20]):  # Limit to 20 for visibility
                mask = values == val
                ax.scatter(coords[mask, 0], coords[mask, 1],
                          c=[palette[i % len(palette)]], label=str(val),
                          alpha=0.5, s=3)

            if legend and len(unique_vals) <= 20:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                         fontsize=8, markerscale=2)
        else:
            # Continuous
            scatter = ax.scatter(coords[:, 0], coords[:, 1],
                                c=values, cmap='viridis',
                                alpha=0.5, s=3)
            plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.5, s=3)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Dimension 1', fontsize=10)
    ax.set_ylabel('Dimension 2', fontsize=10)
    ax.grid(True, alpha=0.3)

# Row 1: UMAP embeddings colored by batch
ax1 = plt.subplot(3, 3, 1)
plot_scatter(ax1, adata.obsm['X_decipher_batch_corrected_umap'],
            batch_key, 'Batch-Corrected UMAP (by Batch)', batch_palette)

ax2 = plt.subplot(3, 3, 2)
if cell_type_key:
    plot_scatter(ax2, adata.obsm['X_decipher_batch_corrected_umap'],
                cell_type_key, 'Batch-Corrected UMAP (by Cell Type)')
else:
    ax2.text(0.5, 0.5, 'No cell type\ncolumn found',
            ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title('Batch-Corrected UMAP (by Cell Type)', fontsize=12, fontweight='bold')

ax3 = plt.subplot(3, 3, 3)
if pseudotime_key:
    plot_scatter(ax3, adata.obsm['X_decipher_batch_corrected_umap'],
                pseudotime_key, 'Batch-Corrected UMAP (by Pseudotime)')
else:
    ax3.text(0.5, 0.5, 'No pseudotime\ncolumn found',
            ha='center', va='center', transform=ax3.transAxes)
    ax3.set_title('Batch-Corrected UMAP (by Pseudotime)', fontsize=12, fontweight='bold')

# Row 2: Latent space (Z, first 2 dimensions)
ax4 = plt.subplot(3, 3, 4)
z_coords = adata.obsm['X_decipher_batch_corrected_z'][:, :2]
plot_scatter(ax4, z_coords, batch_key,
            'Latent Z Space (by Batch)', batch_palette, legend=False)

ax5 = plt.subplot(3, 3, 5)
if cell_type_key:
    plot_scatter(ax5, z_coords, cell_type_key,
                'Latent Z Space (by Cell Type)', legend=False)
else:
    ax5.text(0.5, 0.5, 'No cell type\ncolumn found',
            ha='center', va='center', transform=ax5.transAxes)
    ax5.set_title('Latent Z Space (by Cell Type)', fontsize=12, fontweight='bold')

ax6 = plt.subplot(3, 3, 6)
plot_scatter(ax6, z_coords, 'batch_attention_strength',
            'Latent Z Space (by Attention Strength)')

# Row 3: Component space (V, 2D)
ax7 = plt.subplot(3, 3, 7)
v_coords = adata.obsm['X_decipher_batch_corrected_v']
plot_scatter(ax7, v_coords, batch_key,
            'Component V Space (by Batch)', batch_palette, legend=False)

ax8 = plt.subplot(3, 3, 8)
if cell_type_key:
    plot_scatter(ax8, v_coords, cell_type_key,
                'Component V Space (by Cell Type)', legend=False)
else:
    ax8.text(0.5, 0.5, 'No cell type\ncolumn found',
            ha='center', va='center', transform=ax8.transAxes)
    ax8.set_title('Component V Space (by Cell Type)', fontsize=12, fontweight='bold')

ax9 = plt.subplot(3, 3, 9)
if pseudotime_key:
    plot_scatter(ax9, v_coords, pseudotime_key,
                'Component V Space (by Pseudotime)')
else:
    ax9.text(0.5, 0.5, 'No pseudotime\ncolumn found',
            ha='center', va='center', transform=ax9.transAxes)
    ax9.set_title('Component V Space (by Pseudotime)', fontsize=12, fontweight='bold')

plt.tight_layout()
output1 = 'bonemarrowmap_batch_correction_comparison.png'
plt.savefig(output1, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output1}")
plt.close()

# =====================================================================
# Figure 2: Attention Analysis
# =====================================================================
print("\n  Creating attention analysis plot...")

fig = plt.figure(figsize=(20, 6))

# Panel 1: Attention distribution
ax1 = plt.subplot(1, 3, 1)
ax1.hist(adata.obs['batch_attention_strength'], bins=50, alpha=0.7, edgecolor='black')
ax1.axvline(adata.obs['batch_attention_strength'].mean(), color='red',
           linestyle='--', linewidth=2, label=f"Mean: {adata.obs['batch_attention_strength'].mean():.4f}")
ax1.set_xlabel('Batch Attention Strength', fontsize=12)
ax1.set_ylabel('Number of Cells', fontsize=12)
ax1.set_title('Distribution of Attention Strength', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel 2: Attention by batch (top 20)
ax2 = plt.subplot(1, 3, 2)
batch_attention = adata.obs.groupby(batch_key)['batch_attention_strength'].mean().sort_values(ascending=False).head(20)
batch_attention.plot(kind='barh', ax=ax2, color='steelblue')
ax2.set_xlabel('Mean Attention Strength', fontsize=12)
ax2.set_ylabel('Batch/Donor', fontsize=12)
ax2.set_title('Top 20 Batches by Attention Strength', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

# Panel 3: Attention by cell type (if available)
ax3 = plt.subplot(1, 3, 3)
if cell_type_key:
    type_attention = adata.obs.groupby(cell_type_key)['batch_attention_strength'].mean().sort_values(ascending=False).head(15)
    type_attention.plot(kind='barh', ax=ax3, color='coral')
    ax3.set_xlabel('Mean Attention Strength', fontsize=12)
    ax3.set_ylabel('Cell Type', fontsize=12)
    ax3.set_title('Top 15 Cell Types by Attention', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
else:
    ax3.text(0.5, 0.5, 'No cell type\ncolumn found',
            ha='center', va='center', transform=ax3.transAxes, fontsize=14)
    ax3.set_title('Cell Type Attention', fontsize=14, fontweight='bold')

plt.tight_layout()
output2 = 'bonemarrowmap_attention_analysis.png'
plt.savefig(output2, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output2}")
plt.close()

# =====================================================================
# Figure 3: Biological Signal Preservation
# =====================================================================
if cell_type_key:
    print("\n  Creating biological signal preservation plot...")

    fig = plt.figure(figsize=(20, 12))

    # Compute cell type purity in neighborhoods
    print("    Computing cell type purity...")
    from sklearn.neighbors import NearestNeighbors

    # Use batch-corrected embeddings
    nbrs = NearestNeighbors(n_neighbors=30, algorithm='auto').fit(
        adata.obsm['X_decipher_batch_corrected_z']
    )
    distances, indices = nbrs.kneighbors(adata.obsm['X_decipher_batch_corrected_z'])

    # Calculate purity: fraction of neighbors with same cell type
    purities = []
    for i, neighbors in enumerate(indices):
        cell_type = adata.obs[cell_type_key].iloc[i]
        neighbor_types = adata.obs[cell_type_key].iloc[neighbors]
        purity = (neighbor_types == cell_type).mean()
        purities.append(purity)

    adata.obs['cell_type_purity'] = purities

    # Panel 1: UMAP colored by purity
    ax1 = plt.subplot(2, 2, 1)
    plot_scatter(ax1, adata.obsm['X_decipher_batch_corrected_umap'],
                'cell_type_purity', 'UMAP by Cell Type Purity')

    # Panel 2: Purity distribution
    ax2 = plt.subplot(2, 2, 2)
    ax2.hist(purities, bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(purities), color='red', linestyle='--',
               linewidth=2, label=f"Mean: {np.mean(purities):.3f}")
    ax2.set_xlabel('Cell Type Purity', fontsize=12)
    ax2.set_ylabel('Number of Cells', fontsize=12)
    ax2.set_title('Distribution of Cell Type Purity', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Cell type counts
    ax3 = plt.subplot(2, 2, 3)
    type_counts = adata.obs[cell_type_key].value_counts().head(20)
    type_counts.plot(kind='barh', ax=ax3, color='teal')
    ax3.set_xlabel('Number of Cells', fontsize=12)
    ax3.set_ylabel('Cell Type', fontsize=12)
    ax3.set_title('Top 20 Cell Types by Count', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')

    # Panel 4: Metrics summary
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')

    metrics_text = f"""
    BIOLOGICAL SIGNAL PRESERVATION METRICS

    Cell Type Purity:
      Mean: {np.mean(purities):.4f}
      Std:  {np.std(purities):.4f}
      Interpretation: Fraction of k=30 nearest neighbors
                     with the same cell type

    Batch Mixing:
      Silhouette score: {batch_corrected_silhouette:.4f}
      Interpretation: Lower is better mixing
                     (range: -1 to 1)

    Dataset Summary:
      Cells: {adata.shape[0]:,}
      Genes: {adata.shape[1]:,}
      Batches: {adata.obs[batch_key].nunique()}
      Cell Types: {adata.obs[cell_type_key].nunique()}

    Attention Statistics:
      Mean: {adata.obs['batch_attention_strength'].mean():.4f}
      Std:  {adata.obs['batch_attention_strength'].std():.4f}
      Min:  {adata.obs['batch_attention_strength'].min():.4f}
      Max:  {adata.obs['batch_attention_strength'].max():.4f}
    """

    ax4.text(0.1, 0.95, metrics_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    output3 = 'bonemarrowmap_biological_preservation.png'
    plt.savefig(output3, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output3}")
    plt.close()
else:
    print("\n  Skipping biological signal plot (no cell type column)")
    output3 = None

# Print summary
print("\n" + "=" * 80)
print("VISUALIZATION COMPLETE")
print("=" * 80)

print("\n✓ Generated visualizations:")
print(f"  1. {output1}")
print(f"     - 3×3 grid showing batch correction across representations")
print(f"     - UMAP, latent Z space, component V space")
print(f"     - Colored by batch, cell type, and pseudotime/attention")

print(f"\n  2. {output2}")
print(f"     - Attention analysis")
print(f"     - Distribution, by batch, and by cell type")

if output3:
    print(f"\n  3. {output3}")
    print(f"     - Biological signal preservation")
    print(f"     - Cell type purity, counts, and metrics")

print("\n📊 Key Metrics:")
if batch_corrected_silhouette:
    print(f"  Batch mixing (Silhouette): {batch_corrected_silhouette:.4f} (lower = better)")
if cell_type_key and 'cell_type_purity' in adata.obs:
    print(f"  Cell type purity: {np.mean(purities):.4f} (higher = better)")
print(f"  Mean attention: {adata.obs['batch_attention_strength'].mean():.4f}")

print("\n💡 Interpretation Guide:")
print("  - Batch mixing: Batches should be well-mixed in UMAP")
print("  - Cell type preservation: Cell types should remain clustered")
print("  - Attention: Higher attention = stronger batch effect correction")
print("  - V space: Should show biological trajectory structure")

print("\n" + "=" * 80)
print("✓ ALL DONE!")
print("=" * 80)
