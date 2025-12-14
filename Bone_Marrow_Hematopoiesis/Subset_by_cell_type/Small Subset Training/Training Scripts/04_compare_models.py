"""
Compare Regular Decipher vs Batch-Corrected Decipher models on Erythroid lineage.

This script compares three models:
1. Regular Decipher (Beta=0.1, no batch correction)
2. Batch-Corrected Decipher (Beta=1.0, 2 heads) - Best config
3. Batch-Corrected Decipher (Beta=0.1, 4 heads) - Weak regularization

Metrics computed:
- Spearman correlation for trajectory preservation
- Silhouette score for cell type separation
- Batch entropy for batch mixing

Visualizations:
- Side-by-side V-space plots colored by cell type and donor
- UMAP comparisons
- Quantitative metric comparisons
- Training curves
"""

import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_score
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("COMPARING DECIPHER MODELS - ERYTHROID LINEAGE")
print("="*80)

# Load all three models
print("\n1. Loading trained models...")
adata_regular = sc.read_h5ad("erythroid_regular_decipher.h5ad")
adata_bc_beta1 = sc.read_h5ad("erythroid_batch_corrected_beta1.0_heads2.h5ad")
adata_bc_beta01 = sc.read_h5ad("erythroid_batch_corrected_beta0.1_heads4.h5ad")

print(f"   Regular Decipher: {adata_regular.n_obs} cells")
print(f"   BC Beta=1.0: {adata_bc_beta1.n_obs} cells")
print(f"   BC Beta=0.1: {adata_bc_beta01.n_obs} cells")

# Define erythroid maturation order for trajectory analysis
# BFU-E → CFU-E → Pro-Erythroblast → Basophilic → Polychromatic → Orthochromatic
erythroid_order = {
    'BFU-E': 0,                        # Burst-forming unit-erythroid (earliest)
    'CFU-E': 1,                        # Colony-forming unit-erythroid
    'Pro-Erythroblast': 2,             # Proerythroblast
    'Basophilic Erythroblast': 3,      # Basophilic normoblast
    'Polychromatic Erythroblast': 4,   # Polychromatic normoblast
    'Orthochromatic Erythroblast': 5   # Orthochromatic normoblast (most mature)
}

print("\n2. Computing batch mixing metrics...")

def compute_batch_entropy(adata, embedding_key, batch_key='Donor', k=30):
    """
    Compute batch entropy for each cell based on k-nearest neighbors.
    Higher entropy = better batch mixing.
    
    Returns mean entropy across all cells.
    """
    from sklearn.neighbors import NearestNeighbors
    
    # Get embeddings
    if embedding_key in adata.obsm:
        X = adata.obsm[embedding_key]
    else:
        print(f"   WARNING: {embedding_key} not found, skipping")
        return np.nan
    
    # Get batch labels
    batches = adata.obs[batch_key].values
    unique_batches = np.unique(batches)
    n_batches = len(unique_batches)
    
    # Create batch to index mapping
    batch_to_idx = {b: i for i, b in enumerate(unique_batches)}
    batch_indices = np.array([batch_to_idx[b] for b in batches])
    
    # Find k nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    # Compute entropy for each cell
    entropies = []
    for i in range(len(X)):
        # Get batches of neighbors (excluding self)
        neighbor_batches = batch_indices[indices[i, 1:]]
        
        # Compute batch frequency distribution
        counts = np.bincount(neighbor_batches, minlength=n_batches)
        probs = counts / counts.sum()
        
        # Compute entropy
        # Filter out zero probabilities
        probs = probs[probs > 0]
        ent = -np.sum(probs * np.log(probs))
        entropies.append(ent)
    
    return np.mean(entropies)


def compute_silhouette_celltype(adata, embedding_key, celltype_key='CellType'):
    """
    Compute silhouette score for cell type separation.
    Higher score = better cell type separation.
    """
    if embedding_key in adata.obsm:
        X = adata.obsm[embedding_key]
    else:
        print(f"   WARNING: {embedding_key} not found, skipping")
        return np.nan
    
    labels = adata.obs[celltype_key].values
    
    # Need at least 2 cell types
    if len(np.unique(labels)) < 2:
        return np.nan
    
    return silhouette_score(X, labels)


def compute_trajectory_correlation(adata, embedding_key, celltype_key='CellType', order_dict=None):
    """
    Compute Spearman correlation between first embedding dimension and cell type order.
    Higher correlation = better trajectory preservation.
    """
    if embedding_key not in adata.obsm:
        print(f"   WARNING: {embedding_key} not found, skipping")
        return np.nan
    
    if order_dict is None:
        return np.nan
    
    # Get first dimension of embedding
    v1 = adata.obsm[embedding_key][:, 0]
    
    # Map cell types to ordered values
    celltypes = adata.obs[celltype_key].values
    order_values = np.array([order_dict.get(ct, np.nan) for ct in celltypes])
    
    # Remove any NaN values
    mask = ~np.isnan(order_values)
    if mask.sum() < 10:
        return np.nan
    
    corr, pval = spearmanr(v1[mask], order_values[mask])
    return corr


# Compute metrics for each model
print("\n   Computing for Regular Decipher...")
regular_metrics = {
    'Spearman (trajectory)': compute_trajectory_correlation(adata_regular, 'decipher_v', order_dict=erythroid_order),
    'Silhouette (cell type)': compute_silhouette_celltype(adata_regular, 'decipher_v'),
    'Batch Entropy (mixing)': compute_batch_entropy(adata_regular, 'decipher_v')
}

print("\n   Computing for BC Beta=1.0...")
bc_beta1_metrics = {
    'Spearman (trajectory)': compute_trajectory_correlation(adata_bc_beta1, 'X_decipher_v', order_dict=erythroid_order),
    'Silhouette (cell type)': compute_silhouette_celltype(adata_bc_beta1, 'X_decipher_v'),
    'Batch Entropy (mixing)': compute_batch_entropy(adata_bc_beta1, 'X_decipher_v')
}

print("\n   Computing for BC Beta=0.1...")
bc_beta01_metrics = {
    'Spearman (trajectory)': compute_trajectory_correlation(adata_bc_beta01, 'X_decipher_v', order_dict=erythroid_order),
    'Silhouette (cell type)': compute_silhouette_celltype(adata_bc_beta01, 'X_decipher_v'),
    'Batch Entropy (mixing)': compute_batch_entropy(adata_bc_beta01, 'X_decipher_v')
}

# Print metrics table
print("\n" + "="*80)
print("QUANTITATIVE COMPARISON")
print("="*80)
print("\nMetrics Summary:")
print("-" * 80)

metrics_df = pd.DataFrame({
    'Regular Decipher': regular_metrics,
    'BC Beta=1.0 (2 heads)': bc_beta1_metrics,
    'BC Beta=0.1 (4 heads)': bc_beta01_metrics
}).T

print(metrics_df.to_string())
print("-" * 80)

print("\nMetric Interpretation:")
print("  • Spearman correlation: Measures trajectory preservation (-1 to 1, higher is better)")
print("  • Silhouette score: Measures cell type separation (-1 to 1, higher is better)")
print("  • Batch entropy: Measures batch mixing (0 to log(n_batches), higher is better)")
print(f"    Max entropy for {adata_regular.obs['Donor'].nunique()} batches: {np.log(adata_regular.obs['Donor'].nunique()):.3f}")

# Create comprehensive visualization
print("\n3. Creating comparison visualizations...")

fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

models = [
    ('Regular Decipher\n(Beta=0.1)', adata_regular, 'decipher_v'),
    ('Batch-Corrected\nBeta=1.0, 2 heads', adata_bc_beta1, 'X_decipher_v'),
    ('Batch-Corrected\nBeta=0.1, 4 heads', adata_bc_beta01, 'X_decipher_v')
]

# Define consistent colors for cell types
cell_types = adata_regular.obs['CellType'].astype('category')
colors = plt.cm.viridis(np.linspace(0, 1, len(cell_types.cat.categories)))
color_dict = dict(zip(cell_types.cat.categories, colors))

# Row 1: V-space colored by Cell Type
for col, (title, adata, emb_key) in enumerate(models):
    ax = fig.add_subplot(gs[0, col])
    
    for ct in cell_types.cat.categories:
        mask = adata.obs['CellType'] == ct
        v_coords = adata.obsm[emb_key][mask]
        ax.scatter(v_coords[:, 0], v_coords[:, 1], c=[color_dict[ct]],
                   label=ct, alpha=0.6, s=15)
    
    ax.set_xlabel('V1', fontsize=10)
    ax.set_ylabel('V2', fontsize=10)
    ax.set_title(f'{title}\nColored by Cell Type', fontsize=11, fontweight='bold')
    if col == 2:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)

# Row 2: V-space colored by Donor (first 20)
donor_colors = plt.cm.tab20(np.linspace(0, 1, 20))
for col, (title, adata, emb_key) in enumerate(models):
    ax = fig.add_subplot(gs[1, col])
    
    donors = adata.obs['Donor'].astype('category')
    for i, donor in enumerate(donors.cat.categories[:20]):
        mask = adata.obs['Donor'] == donor
        v_coords = adata.obsm[emb_key][mask]
        ax.scatter(v_coords[:, 0], v_coords[:, 1], c=[donor_colors[i]],
                   alpha=0.4, s=8)
    
    ax.set_xlabel('V1', fontsize=10)
    ax.set_ylabel('V2', fontsize=10)
    ax.set_title(f'{title}\nColored by Donor (20 shown)', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)

# Row 3: UMAP colored by Cell Type
for col, (title, adata, emb_key) in enumerate(models):
    ax = fig.add_subplot(gs[2, col])
    
    for ct in cell_types.cat.categories:
        mask = adata.obs['CellType'] == ct
        umap_coords = adata.obsm['X_umap'][mask]
        ax.scatter(umap_coords[:, 0], umap_coords[:, 1], c=[color_dict[ct]],
                   label=ct, alpha=0.6, s=15)
    
    ax.set_xlabel('UMAP1', fontsize=10)
    ax.set_ylabel('UMAP2', fontsize=10)
    ax.set_title(f'{title}\nUMAP by Cell Type', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)

# Row 4: Metrics comparison bar plot
ax = fig.add_subplot(gs[3, :])

# Prepare data for grouped bar plot
metrics_names = list(regular_metrics.keys())
x = np.arange(len(metrics_names))
width = 0.25

# Normalize metrics to [0, 1] for better visualization
def normalize_metric(values, metric_name):
    if 'Entropy' in metric_name:
        # Normalize by max possible entropy
        max_entropy = np.log(adata_regular.obs['Donor'].nunique())
        return np.array(values) / max_entropy
    else:
        # For Spearman and Silhouette, shift from [-1,1] to [0,1]
        return (np.array(values) + 1) / 2

regular_vals = [regular_metrics[m] for m in metrics_names]
bc_beta1_vals = [bc_beta1_metrics[m] for m in metrics_names]
bc_beta01_vals = [bc_beta01_metrics[m] for m in metrics_names]

bars1 = ax.bar(x - width, [regular_metrics[m] for m in metrics_names], width, 
               label='Regular Decipher', color='steelblue', edgecolor='black')
bars2 = ax.bar(x, [bc_beta1_metrics[m] for m in metrics_names], width,
               label='BC Beta=1.0 (2 heads)', color='coral', edgecolor='black')
bars3 = ax.bar(x + width, [bc_beta01_metrics[m] for m in metrics_names], width,
               label='BC Beta=0.1 (4 heads)', color='lightgreen', edgecolor='black')

ax.set_xlabel('Metric', fontsize=11, fontweight='bold')
ax.set_ylabel('Score', fontsize=11, fontweight='bold')
ax.set_title('Quantitative Comparison Across Models', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([m.split('(')[0].strip() for m in metrics_names], fontsize=10)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Add value labels on bars
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)

autolabel(bars1)
autolabel(bars2)
autolabel(bars3)

plt.savefig('erythroid_model_comparison.png', dpi=150, bbox_inches='tight')
print("   Saved: erythroid_model_comparison.png")

# Create training curves comparison
print("\n4. Creating training curves comparison...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Regular Decipher training curves
ax = axes[0]
if 'decipher_train_loss' in adata_regular.uns:
    train_loss = adata_regular.uns['decipher_train_loss']
    val_loss = adata_regular.uns['decipher_val_loss']
    epochs = range(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, label='Train', linewidth=2, color='steelblue')
    ax.plot(epochs, val_loss, label='Validation', linewidth=2, color='coral')
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Loss', fontsize=11)
ax.set_title('Regular Decipher Training', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# BC Beta=1.0 training curves
ax = axes[1]
if 'decipher_train_loss' in adata_bc_beta1.uns:
    train_loss = adata_bc_beta1.uns['decipher_train_loss']
    val_loss = adata_bc_beta1.uns['decipher_val_loss']
    epochs = range(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, label='Train', linewidth=2, color='steelblue')
    ax.plot(epochs, val_loss, label='Validation', linewidth=2, color='coral')
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Loss', fontsize=11)
ax.set_title('BC Beta=1.0 Training (Early Stop)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# BC Beta=0.1 training curves
ax = axes[2]
if 'decipher_train_loss' in adata_bc_beta01.uns:
    train_loss = adata_bc_beta01.uns['decipher_train_loss']
    val_loss = adata_bc_beta01.uns['decipher_val_loss']
    epochs = range(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, label='Train', linewidth=2, color='steelblue')
    ax.plot(epochs, val_loss, label='Validation', linewidth=2, color='coral')
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Loss', fontsize=11)
ax.set_title('BC Beta=0.1 Training (Early Stop)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('erythroid_training_comparison.png', dpi=150, bbox_inches='tight')
print("   Saved: erythroid_training_comparison.png")

print("\n" + "="*80)
print("✓ COMPARISON COMPLETE!")
print("="*80)
print("\nOutputs:")
print("  1. erythroid_model_comparison.png - Main comparison figure")
print("  2. erythroid_training_comparison.png - Training curves")
print("\nKey Findings:")
print(f"  • Best trajectory preservation: {metrics_df['Spearman (trajectory)'].idxmax()}")
print(f"    (Spearman = {metrics_df['Spearman (trajectory)'].max():.3f})")
print(f"  • Best cell type separation: {metrics_df['Silhouette (cell type)'].idxmax()}")
print(f"    (Silhouette = {metrics_df['Silhouette (cell type)'].max():.3f})")
print(f"  • Best batch mixing: {metrics_df['Batch Entropy (mixing)'].idxmax()}")
print(f"    (Entropy = {metrics_df['Batch Entropy (mixing)'].max():.3f})")

