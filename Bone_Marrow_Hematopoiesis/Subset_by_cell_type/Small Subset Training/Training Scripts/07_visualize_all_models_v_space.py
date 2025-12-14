"""
Create Six V-space Visualizations for All Three Models

This script creates:
1. V-space visualization (3 plots - one for each model)
2. V-space colored by pseudotime (3 plots - one for each model)

Models compared:
- Regular Decipher (no batch correction)
- Attention-based BC (Beta=1.0, 2 heads)
- Concatenation-based BC (Beta=1.0)
"""

import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import warnings
import os

warnings.filterwarnings('ignore')

print("="*80)
print("V-SPACE VISUALIZATION - ALL THREE MODELS")
print("="*80)

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
models_dir = os.path.join(base_dir, "Models")
viz_dir = os.path.join(base_dir, "Visualizations")

# Load all three models
print("\n1. Loading trained models...")
adata_regular = sc.read_h5ad(os.path.join(models_dir, "erythroid_regular_decipher.h5ad"))
adata_attention = sc.read_h5ad(os.path.join(models_dir, "erythroid_batch_corrected_beta1.0_heads2.h5ad"))
adata_concat = sc.read_h5ad(os.path.join(models_dir, "erythroid_concat_beta1.0.h5ad"))

print(f"   Regular Decipher: {adata_regular.n_obs} cells")
print(f"   Attention BC (Beta=1.0, 2 heads): {adata_attention.n_obs} cells")
print(f"   Concatenation BC (Beta=1.0): {adata_concat.n_obs} cells")

# Define erythroid maturation order
erythroid_order = {
    'BFU-E': 0,
    'CFU-E': 1,
    'Pro-Erythroblast': 2,
    'Basophilic Erythroblast': 3,
    'Polychromatic Erythroblast': 4,
    'Orthochromatic Erythroblast': 5
}

print("\n2. Computing pseudotime for all models...")

def compute_diffusion_pseudotime(adata, embedding_key, root_type='BFU-E'):
    """
    Compute diffusion pseudotime in V-space.
    Uses diffusion maps to capture the continuous progression.
    """
    # Create a temporary copy for pseudotime calculation
    adata_temp = adata.copy()

    # Set the embedding as X for neighbor calculation
    adata_temp.obsm['X_embedding'] = adata_temp.obsm[embedding_key]

    # Compute neighbors in V-space
    sc.pp.neighbors(adata_temp, use_rep='X_embedding', n_neighbors=30)

    # Compute diffusion map
    sc.tl.diffmap(adata_temp)

    # Find root cell (use centroid of earliest cell type)
    root_mask = adata_temp.obs['CellType'] == root_type
    if root_mask.sum() > 0:
        # Use the cell closest to the centroid of root cell type
        root_cells = np.where(root_mask)[0]
        root_centroid = adata_temp.obsm['X_embedding'][root_cells].mean(axis=0)
        distances = np.linalg.norm(
            adata_temp.obsm['X_embedding'][root_cells] - root_centroid,
            axis=1
        )
        root_cell = root_cells[np.argmin(distances)]
    else:
        root_cell = 0

    # Compute DPT
    adata_temp.uns['iroot'] = root_cell
    sc.tl.dpt(adata_temp)

    return adata_temp.obs['dpt_pseudotime'].values

# Compute pseudotime for each model
print("   Computing for Regular Decipher...")
adata_regular.obs['pseudotime'] = compute_diffusion_pseudotime(adata_regular, 'decipher_v')

print("   Computing for Attention BC...")
adata_attention.obs['pseudotime'] = compute_diffusion_pseudotime(adata_attention, 'X_decipher_v')

print("   Computing for Concatenation BC...")
adata_concat.obs['pseudotime'] = compute_diffusion_pseudotime(adata_concat, 'X_decipher_v')

print("\n3. Creating visualizations...")

# Define models
models = [
    ('Regular Decipher', adata_regular, 'decipher_v'),
    ('Attention BC\n(Beta=1.0, 2 heads)', adata_attention, 'X_decipher_v'),
    ('Concatenation BC\n(Beta=1.0)', adata_concat, 'X_decipher_v')
]

# Color scheme for cell types (ordered by maturation)
cell_type_colors = plt.cm.RdYlBu_r(np.linspace(0, 1, 6))
cell_type_dict = dict(zip(sorted(erythroid_order.keys(), key=lambda x: erythroid_order[x]),
                          cell_type_colors))

# Create figure with 2 rows × 3 columns
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)

# Row 1: V-space colored by cell type
for col, (title, adata, emb_key) in enumerate(models):
    ax = fig.add_subplot(gs[0, col])

    v_coords = adata.obsm[emb_key]

    # Plot in order of maturation
    for ct in sorted(erythroid_order.keys(), key=lambda x: erythroid_order[x]):
        mask = adata.obs['CellType'] == ct
        if mask.sum() > 0:
            ax.scatter(v_coords[mask, 0], v_coords[mask, 1],
                      c=[cell_type_dict[ct]], label=ct,
                      alpha=0.6, s=25, edgecolors='none')

    ax.set_xlabel('V1', fontsize=13, fontweight='bold')
    ax.set_ylabel('V2', fontsize=13, fontweight='bold')
    ax.set_title(f'{title}\nV-space by Cell Type', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')

    if col == 2:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11,
                 title='Maturation →', title_fontsize=12)

# Row 2: V-space colored by pseudotime
for col, (title, adata, emb_key) in enumerate(models):
    ax = fig.add_subplot(gs[1, col])

    v_coords = adata.obsm[emb_key]
    pseudotime = adata.obs['pseudotime'].values

    scatter = ax.scatter(v_coords[:, 0], v_coords[:, 1],
                        c=pseudotime, cmap='viridis',
                        alpha=0.7, s=25, edgecolors='none')

    ax.set_xlabel('V1', fontsize=13, fontweight='bold')
    ax.set_ylabel('V2', fontsize=13, fontweight='bold')
    ax.set_title(f'{title}\nV-space by Pseudotime', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Pseudotime (Early → Late)', fontsize=11)

# Add overall title
fig.suptitle('V-space Comparison: Regular vs Attention-based BC vs Concatenation BC',
             fontsize=16, fontweight='bold', y=0.98)

# Save figure
output_file = os.path.join(viz_dir, 'all_models_v_space_comparison.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {output_file}")

print("\n" + "="*80)
print("✓ VISUALIZATION COMPLETE!")
print("="*80)

# Compute and display correlation metrics
print("\n4. Computing V1-Pseudotime correlations...")

correlations = {}
for title, adata, emb_key in models:
    v1 = adata.obsm[emb_key][:, 0]
    pseudotime = adata.obs['pseudotime'].values

    # Remove any NaN values
    mask = ~np.isnan(pseudotime)
    if mask.sum() > 10:
        corr, pval = spearmanr(v1[mask], pseudotime[mask])
        correlations[title.replace('\n', ' ')] = {
            'Spearman ρ': corr,
            'p-value': pval
        }

corr_df = pd.DataFrame(correlations).T
print("\nSpearman Correlation (V1 vs Pseudotime):")
print("-" * 80)
print(corr_df.to_string())
print("-" * 80)

print("\nInterpretation:")
print("  • Higher |ρ| indicates better alignment with developmental trajectory")
print("  • All p-values should be << 0.05 for significant correlation")
print(f"\n  Best model: {corr_df['Spearman ρ'].abs().idxmax()}")
print(f"  (|ρ| = {corr_df['Spearman ρ'].abs().max():.3f})")

print("\nOutputs:")
print(f"  1. {output_file}")
print("\nThe visualization shows:")
print("  - Row 1: V-space colored by cell type (discrete maturation stages)")
print("  - Row 2: V-space colored by pseudotime (continuous developmental trajectory)")
print("  - All three models side-by-side for easy comparison")
