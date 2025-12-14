"""
Compute Comprehensive Metrics Table for All Three Models

This script computes and displays:
1. Silhouette Score (cell type separation in V-space)
2. Spearman Correlation (V1 vs pseudotime alignment)
3. Batch Entropy (donor mixing in V-space)

For all three models:
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
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import warnings
import os

warnings.filterwarnings('ignore')

print("="*80)
print("COMPREHENSIVE METRICS - ALL THREE MODELS")
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

print("\n3. Computing metrics...")

def compute_silhouette_celltype(adata, embedding_key, celltype_key='CellType'):
    """Compute silhouette score for cell type separation in V-space."""
    X = adata.obsm[embedding_key]
    labels = adata.obs[celltype_key].values

    if len(np.unique(labels)) < 2:
        return np.nan

    return silhouette_score(X, labels)


def compute_spearman_pseudotime(adata, embedding_key):
    """
    Compute Spearman correlation between V1 and pseudotime.
    Measures how well V1 captures the developmental trajectory.
    """
    v1 = adata.obsm[embedding_key][:, 0]
    pseudotime = adata.obs['pseudotime'].values

    # Remove any NaN values
    mask = ~np.isnan(pseudotime)
    if mask.sum() < 10:
        return np.nan

    corr, pval = spearmanr(v1[mask], pseudotime[mask])
    return corr


def compute_batch_entropy(adata, embedding_key, batch_key='Donor', k=30):
    """Compute batch entropy for V-space mixing."""
    X = adata.obsm[embedding_key]
    batches = adata.obs[batch_key].values
    unique_batches = np.unique(batches)
    n_batches = len(unique_batches)

    batch_to_idx = {b: i for i, b in enumerate(unique_batches)}
    batch_indices = np.array([batch_to_idx[b] for b in batches])

    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    distances, indices = nbrs.kneighbors(X)

    entropies = []
    for i in range(len(X)):
        neighbor_batches = batch_indices[indices[i, 1:]]
        counts = np.bincount(neighbor_batches, minlength=n_batches)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        ent = -np.sum(probs * np.log(probs))
        entropies.append(ent)

    return np.mean(entropies)


# Compute all metrics for each model
print("   Computing for Regular Decipher...")
regular_metrics = {
    'Silhouette Score': compute_silhouette_celltype(adata_regular, 'decipher_v'),
    'Spearman Coefficient': compute_spearman_pseudotime(adata_regular, 'decipher_v'),
    'Batch Entropy': compute_batch_entropy(adata_regular, 'decipher_v')
}

print("   Computing for Attention BC...")
attention_metrics = {
    'Silhouette Score': compute_silhouette_celltype(adata_attention, 'X_decipher_v'),
    'Spearman Coefficient': compute_spearman_pseudotime(adata_attention, 'X_decipher_v'),
    'Batch Entropy': compute_batch_entropy(adata_attention, 'X_decipher_v')
}

print("   Computing for Concatenation BC...")
concat_metrics = {
    'Silhouette Score': compute_silhouette_celltype(adata_concat, 'X_decipher_v'),
    'Spearman Coefficient': compute_spearman_pseudotime(adata_concat, 'X_decipher_v'),
    'Batch Entropy': compute_batch_entropy(adata_concat, 'X_decipher_v')
}

# Create DataFrame
metrics_df = pd.DataFrame({
    'Regular Decipher': regular_metrics,
    'Attention BC (β=1.0, 2 heads)': attention_metrics,
    'Concatenation BC (β=1.0)': concat_metrics
}).T

print("\n" + "="*80)
print("METRICS COMPARISON TABLE")
print("="*80)
print()
print(metrics_df.to_string(float_format=lambda x: f'{x:.4f}'))
print()
print("="*80)

# Identify best models for each metric
print("\nBEST MODELS PER METRIC:")
print("-" * 80)

# Silhouette Score (higher is better)
best_sil = metrics_df['Silhouette Score'].idxmax()
print(f"Silhouette Score (cell type separation):")
print(f"  Winner: {best_sil}")
print(f"  Score: {metrics_df.loc[best_sil, 'Silhouette Score']:.4f}")
print(f"  → Higher is better (range: -1 to 1)")

# Spearman Coefficient (higher absolute value is better)
best_spear = metrics_df['Spearman Coefficient'].abs().idxmax()
print(f"\nSpearman Coefficient (trajectory alignment):")
print(f"  Winner: {best_spear}")
print(f"  Score: {metrics_df.loc[best_spear, 'Spearman Coefficient']:.4f}")
print(f"  → Higher |ρ| is better (range: -1 to 1)")

# Batch Entropy (higher is better)
best_entropy = metrics_df['Batch Entropy'].idxmax()
max_entropy = np.log(45)  # 45 donors
print(f"\nBatch Entropy (donor mixing):")
print(f"  Winner: {best_entropy}")
print(f"  Score: {metrics_df.loc[best_entropy, 'Batch Entropy']:.4f}")
print(f"  → Higher is better (range: 0 to {max_entropy:.3f})")
print(f"  → Mixing efficiency: {metrics_df.loc[best_entropy, 'Batch Entropy']/max_entropy*100:.1f}%")

print("\n" + "="*80)
print("METRIC INTERPRETATIONS")
print("="*80)

print("""
1. SILHOUETTE SCORE (Cell Type Separation):
   • Measures how well-separated cell types are in V-space
   • Range: -1 (poor) to +1 (excellent)
   • Higher values indicate distinct cell type clusters
   • Trade-off: May conflict with trajectory continuity

2. SPEARMAN COEFFICIENT (Trajectory Alignment):
   • Correlation between V1 dimension and pseudotime
   • Range: -1 to +1 (absolute value matters)
   • Higher |ρ| indicates V1 captures developmental progression
   • Negative/positive just indicates axis direction

3. BATCH ENTROPY (Donor Mixing):
   • Measures diversity of donors in local neighborhoods
   • Range: 0 (no mixing) to log(45)=3.807 (perfect mixing)
   • Higher values indicate better batch correction
   • Goal: Mix donors while preserving biology
""")

# Create comparison metrics table
print("\n" + "="*80)
print("NORMALIZED METRICS (0-100 scale)")
print("="*80)

# Normalize metrics to 0-100 scale for easier interpretation
normalized_df = metrics_df.copy()

# Silhouette: map [-1, 1] to [0, 100]
normalized_df['Silhouette Score'] = (metrics_df['Silhouette Score'] + 1) * 50

# Spearman: use absolute value and map [0, 1] to [0, 100]
normalized_df['Spearman Coefficient'] = metrics_df['Spearman Coefficient'].abs() * 100

# Batch Entropy: map [0, log(45)] to [0, 100]
normalized_df['Batch Entropy'] = (metrics_df['Batch Entropy'] / max_entropy) * 100

print()
print(normalized_df.to_string(float_format=lambda x: f'{x:.1f}'))
print()
print("Note: All metrics scaled to 0-100, where higher is better")
print("="*80)

# Save metrics to CSV
csv_file = os.path.join(base_dir, 'erythroid_metrics_comparison.csv')
metrics_df.to_csv(csv_file)
print(f"\n✓ Metrics saved to: {csv_file}")

# Create visualization
print("\n4. Creating metrics visualization...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

model_names = ['Regular\nDecipher', 'Attention BC\n(β=1.0, 2h)', 'Concat BC\n(β=1.0)']
colors = ['#3498db', '#e74c3c', '#2ecc71']

# Plot 1: Silhouette Score
ax = axes[0]
values = metrics_df['Silhouette Score'].values
bars = ax.bar(model_names, values, color=colors, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
ax.set_title('Cell Type Separation\n(Higher is Better)', fontsize=13, fontweight='bold')
ax.set_ylim([0, max(values) * 1.2])
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
# Add value labels
for i, (bar, val) in enumerate(zip(bars, values)):
    ax.text(bar.get_x() + bar.get_width()/2, val + max(values)*0.02,
            f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# Plot 2: Spearman Coefficient (absolute value)
ax = axes[1]
values = metrics_df['Spearman Coefficient'].abs().values
bars = ax.bar(model_names, values, color=colors, edgecolor='black', linewidth=1.5)
ax.set_ylabel('|Spearman ρ|', fontsize=12, fontweight='bold')
ax.set_title('Trajectory Alignment\n(Higher is Better)', fontsize=13, fontweight='bold')
ax.set_ylim([0, 1.0])
ax.grid(axis='y', alpha=0.3, linestyle='--')
# Add value labels
for i, (bar, val) in enumerate(zip(bars, values)):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
            f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# Plot 3: Batch Entropy
ax = axes[2]
values = metrics_df['Batch Entropy'].values
bars = ax.bar(model_names, values, color=colors, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Batch Entropy', fontsize=12, fontweight='bold')
ax.set_title('Donor Mixing\n(Higher is Better)', fontsize=13, fontweight='bold')
ax.set_ylim([0, max_entropy])
ax.axhline(y=max_entropy, color='red', linestyle='--', linewidth=1.5,
           label=f'Max ({max_entropy:.2f})', alpha=0.7)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.legend(fontsize=10)
# Add value labels
for i, (bar, val) in enumerate(zip(bars, values)):
    ax.text(bar.get_x() + bar.get_width()/2, val + max_entropy*0.02,
            f'{val:.3f}\n({val/max_entropy*100:.0f}%)', ha='center', va='bottom',
            fontweight='bold', fontsize=10)

plt.suptitle('V-space Metrics Comparison: Erythroid Subset',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

plot_file = os.path.join(viz_dir, 'erythroid_metrics_comparison.png')
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved to: {plot_file}")

print("\n" + "="*80)
print("✓ METRICS COMPUTATION COMPLETE!")
print("="*80)

print("\nOutputs:")
print(f"  1. {csv_file}")
print(f"  2. {plot_file}")

print("\nSummary:")
print(f"  • Best cell type separation: {best_sil}")
print(f"  • Best trajectory alignment: {best_spear}")
print(f"  • Best donor mixing: {best_entropy}")
