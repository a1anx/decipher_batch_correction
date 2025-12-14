"""
Create side-by-side comparison of Regular Decipher vs Batch-Corrected Decipher

This script:
1. Loads both regular and batch-corrected Decipher results
2. Calculates pseudotime for both using diffusion pseudotime (DPT)
3. Creates side-by-side 3×3 comparison grids
4. Colors V space by pseudotime

Usage:
    python create_decipher_comparison.py [--epochs EPOCHS]
"""

import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import argparse
import os

# Parse arguments
parser = argparse.ArgumentParser(description='Create Decipher comparison visualization')
parser.add_argument('--epochs', type=int, default=None,
                   help='Number of epochs used in batch-corrected training (for file naming)')
args = parser.parse_args()

# Set file paths and output directory
if args.epochs and args.epochs >= 100:
    bc_file = f"Full_Training/bonemarrowmap_small_{args.epochs}epochs_batch_corrected.h5ad"
    output_file = f"bonemarrowmap_small_{args.epochs}epochs_decipher_comparison.png"
    output_dir = "Full_Training"
    epochs_desc = f" ({args.epochs} Epochs)"
else:
    bc_file = "bonemarrowmap_small_batch_corrected.h5ad"
    output_file = "bonemarrowmap_small_decipher_comparison.png"
    output_dir = "."
    epochs_desc = ""

print("=" * 80)
print(f"DECIPHER COMPARISON: Regular vs Batch-Corrected{epochs_desc}")
print("=" * 80)

# Load both datasets
print("\n[1/5] Loading data...")
adata_regular = sc.read_h5ad("bonemarrowmap_small_regular_decipher.h5ad")
adata_bc = sc.read_h5ad(bc_file)
print(f"✓ Regular Decipher: {adata_regular.shape[0]:,} cells")
print(f"✓ Batch-corrected Decipher: {adata_bc.shape[0]:,} cells")

# Detect metadata columns
batch_cols = [col for col in adata_bc.obs.columns
              if 'donor' in col.lower() or 'batch' in col.lower()]
batch_key = batch_cols[0] if batch_cols else None

cell_type_cols = [col for col in adata_bc.obs.columns
                  if 'celltype' in col.lower() or 'cell_type' in col.lower()]
cell_type_key = cell_type_cols[0] if cell_type_cols else None

print(f"  Batch column: '{batch_key}'")
print(f"  Cell type column: '{cell_type_key}'")

# Calculate pseudotime for both models
print("\n[2/5] Calculating pseudotime...")

def calculate_pseudotime(adata, embedding_key, model_name):
    """Calculate diffusion pseudotime on the given embedding"""
    print(f"\n  [{model_name}] Computing pseudotime...")

    # Use the latent z space for pseudotime calculation
    adata.obsm['X_pca'] = adata.obsm[embedding_key]

    # Compute neighbors
    sc.pp.neighbors(adata, use_rep='X_pca', n_neighbors=30)

    # Find root cell (use HSC-like cell type if available, else use first cell)
    if cell_type_key and cell_type_key in adata.obs.columns:
        # Look for HSC or progenitor cell types
        hsc_types = [ct for ct in adata.obs[cell_type_key].unique()
                     if any(keyword in str(ct).lower() for keyword in ['hsc', 'stem', 'progenitor', 'prog'])]
        if hsc_types:
            root_cells = adata.obs[adata.obs[cell_type_key] == hsc_types[0]].index
            root_cell_idx = adata.obs.index.get_loc(root_cells[0])
            print(f"    Using root cell type: {hsc_types[0]}")
        else:
            root_cell_idx = 0
            print(f"    Using first cell as root")
    else:
        root_cell_idx = 0
        print(f"    Using first cell as root")

    # Compute diffusion pseudotime
    adata.uns['iroot'] = root_cell_idx
    sc.tl.diffmap(adata)
    sc.tl.dpt(adata)

    print(f"    ✓ Pseudotime range: [{adata.obs['dpt_pseudotime'].min():.3f}, {adata.obs['dpt_pseudotime'].max():.3f}]")

    return adata.obs['dpt_pseudotime'].values

# Calculate pseudotime for both
pt_regular = calculate_pseudotime(adata_regular, 'X_decipher_regular_z', 'Regular')
pt_bc = calculate_pseudotime(adata_bc, 'X_decipher_batch_corrected_z', 'Batch-Corrected')

# Add pseudotime to obs
adata_regular.obs['pseudotime'] = pt_regular
adata_bc.obs['pseudotime'] = pt_bc

# Create side-by-side comparison
print("\n[3/5] Creating comparison visualization...")

fig = plt.figure(figsize=(20, 12))
gs = GridSpec(3, 6, figure=fig, hspace=0.3, wspace=0.4)

# Color maps
batch_colors = sns.color_palette("tab20", n_colors=min(20, adata_bc.obs[batch_key].nunique()))
celltype_colors = sns.color_palette("tab20", n_colors=min(20, adata_bc.obs[cell_type_key].nunique()))

# Helper function to plot
def plot_embedding(ax, adata, embedding_key, color_by, title, cmap=None, palette=None):
    """Plot embedding colored by specified column"""
    coords = adata.obsm[embedding_key]

    if color_by == 'batch':
        if batch_key:
            batches = adata.obs[batch_key]
            unique_batches = batches.unique()
            for i, batch in enumerate(unique_batches):
                mask = batches == batch
                ax.scatter(coords[mask, 0], coords[mask, 1],
                          c=[batch_colors[i % len(batch_colors)]],
                          s=5, alpha=0.6, label=str(batch) if i < 10 else "")
            if len(unique_batches) <= 10:
                ax.legend(markerscale=2, fontsize=6, loc='best', framealpha=0.5)
    elif color_by == 'celltype':
        if cell_type_key:
            celltypes = adata.obs[cell_type_key]
            unique_types = celltypes.unique()
            for i, ct in enumerate(unique_types):
                mask = celltypes == ct
                ax.scatter(coords[mask, 0], coords[mask, 1],
                          c=[celltype_colors[i % len(celltype_colors)]],
                          s=5, alpha=0.6, label=str(ct) if i < 10 else "")
            if len(unique_types) <= 10:
                ax.legend(markerscale=2, fontsize=6, loc='best', framealpha=0.5)
    elif color_by == 'pseudotime':
        sc = ax.scatter(coords[:, 0], coords[:, 1],
                       c=adata.obs['pseudotime'],
                       s=5, alpha=0.6, cmap='viridis')
        plt.colorbar(sc, ax=ax, label='Pseudotime', fraction=0.046, pad=0.04)
    elif color_by == 'attention':
        if 'batch_attention_strength' in adata.obs:
            sc = ax.scatter(coords[:, 0], coords[:, 1],
                           c=adata.obs['batch_attention_strength'],
                           s=5, alpha=0.6, cmap='Reds')
            plt.colorbar(sc, ax=ax, label='Attention', fraction=0.046, pad=0.04)

    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('Dim 1', fontsize=8)
    ax.set_ylabel('Dim 2', fontsize=8)
    ax.tick_params(labelsize=6)

# Row 1: UMAP embeddings
row_labels = ['UMAP Space', 'Latent Z Space', 'Component V Space']
col_labels_left = ['By Batch', 'By Cell Type', 'By Pseudotime']
col_labels_right = ['By Batch', 'By Cell Type', 'By Pseudotime/Attention']

# Regular Decipher (left column)
print("  Plotting Regular Decipher...")
# UMAP
ax = fig.add_subplot(gs[0, 0])
plot_embedding(ax, adata_regular, 'X_decipher_regular_umap', 'batch',
              f'Regular Decipher - {row_labels[0]}\n{col_labels_left[0]}')

ax = fig.add_subplot(gs[0, 1])
plot_embedding(ax, adata_regular, 'X_decipher_regular_umap', 'celltype',
              f'{col_labels_left[1]}')

ax = fig.add_subplot(gs[0, 2])
plot_embedding(ax, adata_regular, 'X_decipher_regular_umap', 'pseudotime',
              f'{col_labels_left[2]}')

# Latent Z (project to 2D using PCA)
if adata_regular.obsm['X_decipher_regular_z'].shape[1] > 2:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(adata_regular.obsm['X_decipher_regular_z'])
    adata_regular.obsm['X_decipher_regular_z_2d'] = z_2d
    z_key = 'X_decipher_regular_z_2d'
else:
    z_key = 'X_decipher_regular_z'

ax = fig.add_subplot(gs[1, 0])
plot_embedding(ax, adata_regular, z_key, 'batch',
              f'{row_labels[1]}\n{col_labels_left[0]}')

ax = fig.add_subplot(gs[1, 1])
plot_embedding(ax, adata_regular, z_key, 'celltype',
              f'{col_labels_left[1]}')

ax = fig.add_subplot(gs[1, 2])
plot_embedding(ax, adata_regular, z_key, 'pseudotime',
              f'{col_labels_left[2]}')

# Component V
ax = fig.add_subplot(gs[2, 0])
plot_embedding(ax, adata_regular, 'X_decipher_regular_v', 'batch',
              f'{row_labels[2]}\n{col_labels_left[0]}')

ax = fig.add_subplot(gs[2, 1])
plot_embedding(ax, adata_regular, 'X_decipher_regular_v', 'celltype',
              f'{col_labels_left[1]}')

ax = fig.add_subplot(gs[2, 2])
plot_embedding(ax, adata_regular, 'X_decipher_regular_v', 'pseudotime',
              f'{col_labels_left[2]}')

# Batch-Corrected Decipher (right column)
print("  Plotting Batch-Corrected Decipher...")
# UMAP
ax = fig.add_subplot(gs[0, 3])
plot_embedding(ax, adata_bc, 'X_decipher_batch_corrected_umap', 'batch',
              f'Batch-Corrected Decipher - {row_labels[0]}\n{col_labels_right[0]}')

ax = fig.add_subplot(gs[0, 4])
plot_embedding(ax, adata_bc, 'X_decipher_batch_corrected_umap', 'celltype',
              f'{col_labels_right[1]}')

ax = fig.add_subplot(gs[0, 5])
plot_embedding(ax, adata_bc, 'X_decipher_batch_corrected_umap', 'pseudotime',
              f'By Pseudotime')

# Latent Z
if adata_bc.obsm['X_decipher_batch_corrected_z'].shape[1] > 2:
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(adata_bc.obsm['X_decipher_batch_corrected_z'])
    adata_bc.obsm['X_decipher_batch_corrected_z_2d'] = z_2d
    z_key_bc = 'X_decipher_batch_corrected_z_2d'
else:
    z_key_bc = 'X_decipher_batch_corrected_z'

ax = fig.add_subplot(gs[1, 3])
plot_embedding(ax, adata_bc, z_key_bc, 'batch',
              f'{row_labels[1]}\n{col_labels_right[0]}')

ax = fig.add_subplot(gs[1, 4])
plot_embedding(ax, adata_bc, z_key_bc, 'celltype',
              f'{col_labels_right[1]}')

ax = fig.add_subplot(gs[1, 5])
plot_embedding(ax, adata_bc, z_key_bc, 'pseudotime',
              f'By Pseudotime')

# Component V
ax = fig.add_subplot(gs[2, 3])
plot_embedding(ax, adata_bc, 'X_decipher_batch_corrected_v', 'batch',
              f'{row_labels[2]}\n{col_labels_right[0]}')

ax = fig.add_subplot(gs[2, 4])
plot_embedding(ax, adata_bc, 'X_decipher_batch_corrected_v', 'celltype',
              f'{col_labels_right[1]}')

ax = fig.add_subplot(gs[2, 5])
plot_embedding(ax, adata_bc, 'X_decipher_batch_corrected_v', 'pseudotime',
              f'By Pseudotime')

# Add overall title
title_text = f'Decipher Comparison: Regular vs Batch-Corrected{epochs_desc}\nBoneMarrowMap SMALL Subset (20k cells)'
fig.suptitle(title_text, fontsize=14, fontweight='bold', y=0.995)

# Save
output_path = os.path.join(output_dir, output_file)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved comparison plot: {output_path}")
plt.close()

# Save updated datasets with pseudotime
print("\n[4/5] Saving datasets with pseudotime...")
adata_regular.write_h5ad("bonemarrowmap_small_regular_decipher.h5ad")
adata_bc.write_h5ad(bc_file)
print("✓ Updated datasets saved")

# Print summary metrics
print("\n[5/5] Computing comparison metrics...")

def compute_mixing_score(adata, embedding_key, batch_key):
    """Compute batch mixing using silhouette score"""
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import LabelEncoder

    X = adata.obsm[embedding_key]
    labels = LabelEncoder().fit_transform(adata.obs[batch_key])

    # For batch correction, lower (more negative) is better
    score = silhouette_score(X, labels)
    return score

# Compute metrics
print("\nBatch Mixing (Silhouette Score - lower is better):")
print(f"  Regular Decipher UMAP:         {compute_mixing_score(adata_regular, 'X_decipher_regular_umap', batch_key):.4f}")
print(f"  Batch-Corrected Decipher UMAP: {compute_mixing_score(adata_bc, 'X_decipher_batch_corrected_umap', batch_key):.4f}")
print(f"  Regular Decipher V-space:         {compute_mixing_score(adata_regular, 'X_decipher_regular_v', batch_key):.4f}")
print(f"  Batch-Corrected Decipher V-space: {compute_mixing_score(adata_bc, 'X_decipher_batch_corrected_v', batch_key):.4f}")

print("\n" + "=" * 80)
print("COMPARISON COMPLETE")
print("=" * 80)
print(f"\n✓ Generated comparison visualization: {output_file}")
print("\n💡 Interpretation:")
print("  - Batch mixing: More negative = better batch mixing")
print("  - V-space with pseudotime: Shows biological trajectory structure")
print("  - Compare batch separation between Regular and Batch-Corrected")
print("\n" + "=" * 80)
