"""
Compute Comprehensive Metrics Table for Three Attention-Based Models (20K Dataset)

This script computes and displays:
1. Silhouette Score (cell type separation in V-space)
2. Spearman Correlation (V1 vs pseudotime alignment)
3. Batch Entropy (donor mixing in V-space)

For three attention-based models trained on 20K dataset:
- Beta=0.1, Attention Heads=4 (Initial Training)
- Beta=1.0, Attention Heads=4
- Beta=1.0, Attention Heads=2
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
print("COMPREHENSIVE METRICS - THREE ATTENTION-BASED MODELS (20K DATASET)")
print("="*80)

# Define paths
base_dir = "/home/alan/Documents/decipher_batch_correction/Bone_Marrow_Hematopoiesis"

# Load all three models
print("\n1. Loading trained models...")
adata_beta01_heads4 = sc.read_h5ad(os.path.join(base_dir, "Initial_Training/bonemarrowmap_small_batch_corrected.h5ad"))
adata_beta10_heads4 = sc.read_h5ad(os.path.join(base_dir, "Beta_1.0_Training/bonemarrowmap_small_150epochs_beta1.h5ad"))
adata_beta10_heads2 = sc.read_h5ad(os.path.join(base_dir, "Beta_1.0_AttentionHeads_2_Training/bonemarrowmap_small_150epochs_beta1_attnheads2.h5ad"))

print(f"   Beta=0.1, Heads=4: {adata_beta01_heads4.n_obs} cells")
print(f"   Beta=1.0, Heads=4: {adata_beta10_heads4.n_obs} cells")
print(f"   Beta=1.0, Heads=2: {adata_beta10_heads2.n_obs} cells")

# Check available embeddings
print("\n2. Checking available embeddings...")
for name, adata in [("Beta=0.1, Heads=4", adata_beta01_heads4),
                     ("Beta=1.0, Heads=4", adata_beta10_heads4),
                     ("Beta=1.0, Heads=2", adata_beta10_heads2)]:
    print(f"\n   {name}:")
    print(f"   obsm keys: {list(adata.obsm.keys())}")
    print(f"   obs keys: {list(adata.obs.keys())[:10]}...")

print("\n3. Computing pseudotime for all models...")

def compute_diffusion_pseudotime(adata, embedding_key, root_type='HSC'):
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
    # Try HSC first, if not available use the first available cell type
    if 'CellType' in adata_temp.obs.columns:
        celltype_col = 'CellType'
    elif 'CellType_Broad' in adata_temp.obs.columns:
        celltype_col = 'CellType_Broad'
    else:
        # Just use first cell if no cell type info
        adata_temp.uns['iroot'] = 0
        sc.tl.dpt(adata_temp)
        return adata_temp.obs['dpt_pseudotime'].values

    # Try to find root cell type
    unique_types = adata_temp.obs[celltype_col].unique()
    if root_type in unique_types:
        root_mask = adata_temp.obs[celltype_col] == root_type
    else:
        # Use first cell type if HSC not found
        root_type = unique_types[0]
        root_mask = adata_temp.obs[celltype_col] == root_type
        print(f"      Warning: Using {root_type} as root (HSC not found)")

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

# Compute pseudotime for each model (using batch-corrected V-space embedding)
print("   Computing for Beta=0.1, Heads=4...")
adata_beta01_heads4.obs['pseudotime'] = compute_diffusion_pseudotime(adata_beta01_heads4, 'X_decipher_batch_corrected_v')

print("   Computing for Beta=1.0, Heads=4...")
adata_beta10_heads4.obs['pseudotime'] = compute_diffusion_pseudotime(adata_beta10_heads4, 'X_decipher_batch_corrected_v')

print("   Computing for Beta=1.0, Heads=2...")
adata_beta10_heads2.obs['pseudotime'] = compute_diffusion_pseudotime(adata_beta10_heads2, 'X_decipher_batch_corrected_v')

print("\n4. Computing metrics...")

def compute_silhouette_celltype(adata, embedding_key, celltype_key='CellType'):
    """Compute silhouette score for cell type separation in V-space."""
    X = adata.obsm[embedding_key]

    # Try different cell type keys
    if celltype_key in adata.obs.columns:
        labels = adata.obs[celltype_key].values
    elif 'CellType_Broad' in adata.obs.columns:
        labels = adata.obs['CellType_Broad'].values
    else:
        return np.nan

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

    if batch_key not in adata.obs.columns:
        return np.nan

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


# Compute all metrics for each model (using batch-corrected V-space embedding)
print("   Computing for Beta=0.1, Heads=4...")
beta01_heads4_metrics = {
    'V-space Batch Silhouette': compute_silhouette_celltype(adata_beta01_heads4, 'X_decipher_batch_corrected_v'),
    'Batch Entropy': compute_batch_entropy(adata_beta01_heads4, 'X_decipher_batch_corrected_v'),
    'V-Pseudotime Correlation (Spearman)': compute_spearman_pseudotime(adata_beta01_heads4, 'X_decipher_batch_corrected_v')
}

print("   Computing for Beta=1.0, Heads=4...")
beta10_heads4_metrics = {
    'V-space Batch Silhouette': compute_silhouette_celltype(adata_beta10_heads4, 'X_decipher_batch_corrected_v'),
    'Batch Entropy': compute_batch_entropy(adata_beta10_heads4, 'X_decipher_batch_corrected_v'),
    'V-Pseudotime Correlation (Spearman)': compute_spearman_pseudotime(adata_beta10_heads4, 'X_decipher_batch_corrected_v')
}

print("   Computing for Beta=1.0, Heads=2...")
beta10_heads2_metrics = {
    'V-space Batch Silhouette': compute_silhouette_celltype(adata_beta10_heads2, 'X_decipher_batch_corrected_v'),
    'Batch Entropy': compute_batch_entropy(adata_beta10_heads2, 'X_decipher_batch_corrected_v'),
    'V-Pseudotime Correlation (Spearman)': compute_spearman_pseudotime(adata_beta10_heads2, 'X_decipher_batch_corrected_v')
}

# Create DataFrame - note the order matches the screenshot
metrics_df = pd.DataFrame({
    'Initial Training (Beta = 0.1, Heads = 4)': beta01_heads4_metrics,
    'Beta = 1.0, Heads = 4': beta10_heads4_metrics,
    'Beta = 1.0, Heads = 2': beta10_heads2_metrics
}).T

print("\n" + "="*80)
print("METRICS COMPARISON TABLE - 20K DATASET")
print("="*80)
print()
print(metrics_df.to_string(float_format=lambda x: f'{x:.2f}'))
print()
print("="*80)

# Identify best models for each metric
print("\nBEST MODELS PER METRIC:")
print("-" * 80)

# Silhouette Score (higher is better for cell type separation)
best_sil = metrics_df['V-space Batch Silhouette'].idxmax()
print(f"V-space Batch Silhouette (cell type separation):")
print(f"  Winner: {best_sil}")
print(f"  Score: {metrics_df.loc[best_sil, 'V-space Batch Silhouette']:.4f}")
print(f"  → Higher is better (range: -1 to 1)")

# Spearman Coefficient (higher absolute value is better)
best_spear = metrics_df['V-Pseudotime Correlation (Spearman)'].abs().idxmax()
print(f"\nV-Pseudotime Correlation (trajectory alignment):")
print(f"  Winner: {best_spear}")
print(f"  Score: {metrics_df.loc[best_spear, 'V-Pseudotime Correlation (Spearman)']:.4f}")
print(f"  → Higher |ρ| is better (range: -1 to 1)")

# Batch Entropy (higher is better)
best_entropy = metrics_df['Batch Entropy'].idxmax()
# Get number of unique donors from first dataset
n_donors = adata_beta01_heads4.obs['Donor'].nunique()
max_entropy = np.log(n_donors)
print(f"\nBatch Entropy (donor mixing):")
print(f"  Winner: {best_entropy}")
print(f"  Score: {metrics_df.loc[best_entropy, 'Batch Entropy']:.4f}")
print(f"  → Higher is better (range: 0 to {max_entropy:.3f})")
print(f"  → Mixing efficiency: {metrics_df.loc[best_entropy, 'Batch Entropy']/max_entropy*100:.1f}%")

print("\n" + "="*80)
print("METRIC INTERPRETATIONS")
print("="*80)

print(f"""
1. V-SPACE BATCH SILHOUETTE (Cell Type Separation):
   • Measures how well-separated cell types are in V-space
   • Range: -1 (poor) to +1 (excellent)
   • Higher values indicate distinct cell type clusters
   • Trade-off: May conflict with batch mixing

2. V-PSEUDOTIME CORRELATION (Trajectory Alignment):
   • Correlation between V1 dimension and pseudotime
   • Range: -1 to +1 (absolute value matters)
   • Higher |ρ| indicates V1 captures developmental progression
   • Negative/positive just indicates axis direction

3. BATCH ENTROPY (Donor Mixing):
   • Measures diversity of donors in local neighborhoods
   • Range: 0 (no mixing) to log({n_donors})={max_entropy:.3f} (perfect mixing)
   • Higher values indicate better batch correction
   • Goal: Mix donors while preserving biology
""")

# Save metrics to CSV
csv_file = os.path.join(base_dir, '20k_attention_models_metrics_comparison.csv')
metrics_df.to_csv(csv_file)
print(f"\n✓ Metrics saved to: {csv_file}")

# Create visualization
print("\n5. Creating metrics visualization...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

model_names = ['Beta=0.1\nHeads=4', 'Beta=1.0\nHeads=4', 'Beta=1.0\nHeads=2']
colors = ['#3498db', '#e74c3c', '#2ecc71']

# Plot 1: Silhouette Score
ax = axes[0]
values = metrics_df['V-space Batch Silhouette'].values
bars = ax.bar(model_names, values, color=colors, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
ax.set_title('Cell Type Separation\n(Higher is Better)', fontsize=13, fontweight='bold')
ax.set_ylim([min(values) * 1.2 if min(values) < 0 else 0, max(values) * 1.2])
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
# Add value labels
for i, (bar, val) in enumerate(zip(bars, values)):
    ax.text(bar.get_x() + bar.get_width()/2, val + (max(values) - min(values))*0.02,
            f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# Plot 2: Spearman Coefficient (absolute value)
ax = axes[1]
values = metrics_df['V-Pseudotime Correlation (Spearman)'].abs().values
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

plt.suptitle('V-space Metrics Comparison: 20K Dataset - Three Attention-Based Models',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

plot_file = os.path.join(base_dir, '20k_attention_models_metrics_comparison.png')
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
