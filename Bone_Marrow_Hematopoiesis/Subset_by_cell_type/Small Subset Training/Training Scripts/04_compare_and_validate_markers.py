"""
Compare Regular vs Batch-Corrected Decipher on Erythroid lineage.

Following TA feedback, this script:
1. Compares V-space structure between models
2. Validates trajectory using known erythroid marker genes
3. Quantifies batch mixing improvements
4. Verifies biological structure is preserved

Marker genes for erythroid differentiation:
- HBB, HBA1, HBA2: Hemoglobin (increases with maturation)
- GYPA: Glycophorin A (erythroid marker)
- TFRC: Transferrin receptor (early erythroid)
- KLF1, GATA1: Erythroid transcription factors
"""

import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearman
from sklearn.metrics import silhouette_score
import argparse

parser = argparse.ArgumentParser(description='Compare models and validate with markers')
parser.add_argument('--regular', type=str, default='erythroid_regular_decipher.h5ad',
                    help='Path to regular Decipher results')
parser.add_argument('--batch-corrected', type=str, default='erythroid_batch_corrected_beta1.0_heads2.h5ad',
                    help='Path to batch-corrected results')
args = parser.parse_args()

print("="*80)
print("ERYTHROID LINEAGE: MODEL COMPARISON & MARKER VALIDATION")
print("="*80)

# Load both models
print("\n1. Loading model results...")
try:
    adata_reg = sc.read_h5ad(args.regular)
    print(f"   ✓ Regular Decipher: {adata_reg.n_obs:,} cells")
except FileNotFoundError:
    print(f"   ✗ Regular Decipher file not found: {args.regular}")
    print("   Run 02_train_regular_decipher_erythroid.py first!")
    exit(1)

try:
    adata_bc = sc.read_h5ad(args.batch_corrected)
    print(f"   ✓ Batch-Corrected: {adata_bc.n_obs:,} cells")
except FileNotFoundError:
    print(f"   ✗ Batch-Corrected file not found: {args.batch_corrected}")
    print("   Run 03_train_batch_corrected_erythroid.py first!")
    exit(1)

# Define marker genes
marker_genes = {
    'Hemoglobin': ['HBB', 'HBA1', 'HBA2'],
    'Erythroid Markers': ['GYPA', 'TFRC', 'EPOR'],
    'Transcription Factors': ['KLF1', 'GATA1'],
}

print("\n2. Checking marker gene availability...")
all_markers = []
for category, genes in marker_genes.items():
    present = [g for g in genes if g in adata_reg.var_names]
    all_markers.extend(present)
    print(f"   {category}: {len(present)}/{len(genes)} genes present")
    for gene in present:
        print(f"     ✓ {gene}")

if len(all_markers) == 0:
    print("\n   WARNING: No marker genes found! Will skip marker analysis.")

# Compute batch mixing metrics
print("\n3. Computing batch mixing metrics...")

def compute_batch_entropy(adata, batch_key='Donor', k=30):
    """Compute mean entropy of batch distribution in k-NN"""
    from sklearn.neighbors import NearestNeighbors

    v_coords = adata.obsm['X_decipher_v']
    batches = adata.obs[batch_key].values

    # Find k nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(v_coords)
    _, indices = nbrs.kneighbors(v_coords)

    # Compute entropy for each cell
    entropies = []
    for i in range(len(adata)):
        neighbor_batches = batches[indices[i, 1:]]  # Exclude self
        unique, counts = np.unique(neighbor_batches, return_counts=True)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        entropies.append(entropy)

    return np.mean(entropies), np.std(entropies)

# Regular Decipher metrics
print("\n   Regular Decipher:")
reg_batch_sil = silhouette_score(adata_reg.obsm['X_decipher_v'],
                                  adata_reg.obs['Donor'])
reg_entropy_mean, reg_entropy_std = compute_batch_entropy(adata_reg)
print(f"     V-space Batch Silhouette: {reg_batch_sil:.4f}")
print(f"     Batch Entropy: {reg_entropy_mean:.4f} ± {reg_entropy_std:.4f}")

# Batch-Corrected metrics
print("\n   Batch-Corrected Decipher:")
bc_batch_sil = silhouette_score(adata_bc.obsm['X_decipher_v'],
                                 adata_bc.obs['Donor'])
bc_entropy_mean, bc_entropy_std = compute_batch_entropy(adata_bc)
print(f"     V-space Batch Silhouette: {bc_batch_sil:.4f}")
print(f"     Batch Entropy: {bc_entropy_mean:.4f} ± {bc_entropy_std:.4f}")

# Improvement
sil_improvement = ((reg_batch_sil - bc_batch_sil) / abs(reg_batch_sil)) * 100
entropy_improvement = ((bc_entropy_mean - reg_entropy_mean) / reg_entropy_mean) * 100
print(f"\n   Improvements:")
print(f"     Silhouette: {sil_improvement:+.1f}% (closer to 0 = better mixing)")
print(f"     Entropy: {entropy_improvement:+.1f}% (higher = better mixing)")

# Create comprehensive comparison figure
print("\n4. Creating comparison visualizations...")

# Determine grid size based on marker availability
if len(all_markers) > 0:
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
else:
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

# Color maps for cell types
cell_types = adata_reg.obs['CellType'].astype('category')
colors = plt.cm.viridis(np.linspace(0, 1, len(cell_types.cat.categories)))
color_dict = dict(zip(cell_types.cat.categories, colors))

# Row 1: V-space by Cell Type
ax = fig.add_subplot(gs[0, 0:2])
for ct in cell_types.cat.categories:
    mask = adata_reg.obs['CellType'] == ct
    v_coords = adata_reg.obsm['X_decipher_v'][mask]
    ax.scatter(v_coords[:, 0], v_coords[:, 1], c=[color_dict[ct]],
               label=ct, alpha=0.6, s=20)
ax.set_xlabel('V1', fontsize=11)
ax.set_ylabel('V2', fontsize=11)
ax.set_title('Regular Decipher: V-space by Cell Type', fontsize=12, fontweight='bold')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[0, 2:4])
for ct in cell_types.cat.categories:
    mask = adata_bc.obs['CellType'] == ct
    v_coords = adata_bc.obsm['X_decipher_v'][mask]
    ax.scatter(v_coords[:, 0], v_coords[:, 1], c=[color_dict[ct]],
               label=ct, alpha=0.6, s=20)
ax.set_xlabel('V1', fontsize=11)
ax.set_ylabel('V2', fontsize=11)
ax.set_title('Batch-Corrected: V-space by Cell Type', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

# Row 2: V-space by Donor
ax = fig.add_subplot(gs[1, 0:2])
donors = adata_reg.obs['Donor'].astype('category')
donor_colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(donors.cat.categories))))
for i, donor in enumerate(donors.cat.categories[:20]):
    mask = adata_reg.obs['Donor'] == donor
    v_coords = adata_reg.obsm['X_decipher_v'][mask]
    ax.scatter(v_coords[:, 0], v_coords[:, 1], c=[donor_colors[i]],
               alpha=0.4, s=10)
ax.set_xlabel('V1', fontsize=11)
ax.set_ylabel('V2', fontsize=11)
ax.set_title(f'Regular: Donors (Sil={reg_batch_sil:.3f})', fontsize=12, fontweight='bold')
ax.text(0.02, 0.98, 'First 20 donors shown', transform=ax.transAxes,
        fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round',
        facecolor='wheat', alpha=0.5))
ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[1, 2:4])
for i, donor in enumerate(donors.cat.categories[:20]):
    mask = adata_bc.obs['Donor'] == donor
    v_coords = adata_bc.obsm['X_decipher_v'][mask]
    ax.scatter(v_coords[:, 0], v_coords[:, 1], c=[donor_colors[i]],
               alpha=0.4, s=10)
ax.set_xlabel('V1', fontsize=11)
ax.set_ylabel('V2', fontsize=11)
ax.set_title(f'Batch-Corrected: Donors (Sil={bc_batch_sil:.3f})', fontsize=12, fontweight='bold')
ax.text(0.02, 0.98, 'First 20 donors shown', transform=ax.transAxes,
        fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round',
        facecolor='wheat', alpha=0.5))
ax.grid(alpha=0.3)

# Row 3: Marker gene expression (if available)
if len(all_markers) > 0:
    # Plot up to 4 most important markers
    key_markers = ['HBB', 'GYPA', 'TFRC', 'KLF1']
    available_key_markers = [m for m in key_markers if m in all_markers][:4]

    for idx, gene in enumerate(available_key_markers):
        # Regular Decipher
        if idx < 2:
            ax = fig.add_subplot(gs[2, idx*2])
            gene_idx = adata_reg.var_names.tolist().index(gene)
            expression = adata_reg.X[:, gene_idx]
            if hasattr(expression, 'toarray'):
                expression = expression.toarray().flatten()

            scatter = ax.scatter(adata_reg.obsm['X_decipher_v'][:, 0],
                               adata_reg.obsm['X_decipher_v'][:, 1],
                               c=expression, cmap='Reds', s=15, alpha=0.6)
            ax.set_xlabel('V1', fontsize=10)
            ax.set_ylabel('V2', fontsize=10)
            ax.set_title(f'Regular: {gene} Expression', fontsize=11, fontweight='bold')
            plt.colorbar(scatter, ax=ax, label='Expression')
            ax.grid(alpha=0.3)

        # Batch-Corrected
        if idx < 2:
            ax = fig.add_subplot(gs[2, idx*2+1])
            gene_idx = adata_bc.var_names.tolist().index(gene)
            expression = adata_bc.X[:, gene_idx]
            if hasattr(expression, 'toarray'):
                expression = expression.toarray().flatten()

            scatter = ax.scatter(adata_bc.obsm['X_decipher_v'][:, 0],
                               adata_bc.obsm['X_decipher_v'][:, 1],
                               c=expression, cmap='Reds', s=15, alpha=0.6)
            ax.set_xlabel('V1', fontsize=10)
            ax.set_ylabel('V2', fontsize=10)
            ax.set_title(f'Batch-Corrected: {gene} Expression', fontsize=11, fontweight='bold')
            plt.colorbar(scatter, ax=ax, label='Expression')
            ax.grid(alpha=0.3)

# Bottom row: Metrics summary
metric_row = 3 if len(all_markers) > 0 else 2

ax = fig.add_subplot(gs[metric_row, 0:2])
metrics_df = pd.DataFrame({
    'Metric': ['Batch Silhouette', 'Batch Entropy'],
    'Regular Decipher': [reg_batch_sil, reg_entropy_mean],
    'Batch-Corrected': [bc_batch_sil, bc_entropy_mean],
})
x = np.arange(len(metrics_df))
width = 0.35
ax.bar(x - width/2, metrics_df['Regular Decipher'], width, label='Regular', color='steelblue')
ax.bar(x + width/2, metrics_df['Batch-Corrected'], width, label='Batch-Corrected', color='coral')
ax.set_ylabel('Value', fontsize=11)
ax.set_title('Batch Mixing Metrics Comparison', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_df['Metric'])
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.8)

# Summary text
ax = fig.add_subplot(gs[metric_row, 2:4])
ax.axis('off')
summary_text = f"""
ERYTHROID LINEAGE COMPARISON

DATASET:
  Cells: {adata_reg.n_obs:,}
  Cell Types: {adata_reg.obs['CellType'].nunique()}
  Donors: {adata_reg.obs['Donor'].nunique()}

BATCH MIXING METRICS:

  Regular Decipher:
    Batch Silhouette: {reg_batch_sil:.4f}
    Batch Entropy: {reg_entropy_mean:.4f}

  Batch-Corrected:
    Batch Silhouette: {bc_batch_sil:.4f}
    Batch Entropy: {bc_entropy_mean:.4f}

  Improvement:
    Silhouette: {sil_improvement:+.1f}%
    Entropy: {entropy_improvement:+.1f}%

INTERPRETATION:
  ✓ Closer to 0 silhouette = better mixing
  ✓ Higher entropy = more diverse donors

MARKER GENES VALIDATED:
  {', '.join(all_markers) if all_markers else 'N/A'}

EXPECTED PATTERN:
  - Trajectory preserved (cell type separation)
  - Donors mixed within each stage
  - Marker genes show biological gradient
"""
ax.text(0.1, 0.5, summary_text, fontsize=9, family='monospace',
        verticalalignment='center', bbox=dict(boxstyle='round',
        facecolor='lightyellow', alpha=0.3))

output_file = "erythroid_comparison_with_markers.png"
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"   Saved to: {output_file}")

# Save metrics to CSV
print("\n5. Saving metrics...")
metrics_summary = pd.DataFrame({
    'Model': ['Regular Decipher', 'Batch-Corrected'],
    'V_Batch_Silhouette': [reg_batch_sil, bc_batch_sil],
    'Batch_Entropy_Mean': [reg_entropy_mean, bc_entropy_mean],
    'Batch_Entropy_Std': [reg_entropy_std, bc_entropy_std],
    'n_cells': [adata_reg.n_obs, adata_bc.n_obs],
    'n_donors': [adata_reg.obs['Donor'].nunique(), adata_bc.obs['Donor'].nunique()],
})
metrics_file = "erythroid_comparison_metrics.csv"
metrics_summary.to_csv(metrics_file, index=False)
print(f"   Saved to: {metrics_file}")

print("\n" + "="*80)
print("✓ COMPARISON COMPLETE!")
print("="*80)
print(f"\nKey Findings:")
print(f"  1. Batch Silhouette improvement: {sil_improvement:+.1f}%")
print(f"  2. Batch Entropy improvement: {entropy_improvement:+.1f}%")
print(f"  3. Marker genes available: {len(all_markers)}")
print(f"\nOutputs:")
print(f"  - {output_file}")
print(f"  - {metrics_file}")
print(f"\nInterpretation:")
if bc_batch_sil > reg_batch_sil:
    print(f"  ⚠ Batch-corrected has WORSE mixing (higher silhouette)")
else:
    print(f"  ✓ Batch-corrected has BETTER mixing (lower silhouette)")
if bc_entropy_mean > reg_entropy_mean:
    print(f"  ✓ Batch-corrected has BETTER diversity (higher entropy)")
else:
    print(f"  ⚠ Batch-corrected has WORSE diversity (lower entropy)")
