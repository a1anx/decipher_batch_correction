"""
Investigate V-space structure and identify what the "islands" represent.
"""

import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

def investigate_v_space(adata_path, exp_name, output_prefix):
    """Investigate V-space structure."""

    print(f"\n{'='*80}")
    print(f"Investigating: {exp_name}")
    print(f"{'='*80}")

    adata = sc.read_h5ad(adata_path)

    # Get V-space
    if 'X_decipher_batch_corrected_v' in adata.obsm:
        v_space = adata.obsm['X_decipher_batch_corrected_v']
        v_key = 'X_decipher_batch_corrected_v'
    elif 'decipher_v' in adata.obsm:
        v_space = adata.obsm['decipher_v']
        v_key = 'decipher_v'
    else:
        print("No V-space found!")
        return

    # Get batch and cell type info
    batch_key = None
    cell_type_key = None

    for col in adata.obs.columns:
        if 'donor' in col.lower() or 'batch' in col.lower():
            batch_key = col
        if 'celltype' in col.lower() or 'cell_type' in col.lower():
            cell_type_key = col

    print(f"\nBatch key: {batch_key}")
    print(f"Cell type key: {cell_type_key}")

    # Identify "islands" using simple clustering on V-space
    print(f"\nClustering V-space to identify islands...")
    kmeans = KMeans(n_clusters=3, random_state=42)
    v_clusters = kmeans.fit_predict(v_space)

    adata.obs['v_cluster'] = v_clusters.astype(str)

    # Analyze each cluster
    print(f"\nV-space clusters:")
    for cluster_id in range(3):
        mask = v_clusters == cluster_id
        n_cells = mask.sum()

        print(f"\n  Cluster {cluster_id}: {n_cells} cells ({100*n_cells/len(adata):.1f}%)")

        # Batch distribution
        if batch_key:
            batch_dist = adata.obs[batch_key][mask].value_counts().head(5)
            print(f"    Top 5 batches:")
            for batch, count in batch_dist.items():
                pct = 100 * count / n_cells
                print(f"      {str(batch)[:30]:30s}: {count:5d} ({pct:5.1f}%)")

        # Cell type distribution
        if cell_type_key:
            cell_type_dist = adata.obs[cell_type_key][mask].value_counts().head(5)
            print(f"    Top 5 cell types:")
            for ct, count in cell_type_dist.items():
                pct = 100 * count / n_cells
                print(f"      {str(ct)[:30]:30s}: {count:5d} ({pct:5.1f}%)")

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Get pseudotime if available
    pseudotime = adata.obs['dpt_pseudotime'] if 'dpt_pseudotime' in adata.obs else None

    # Plot 1: V-space by cluster
    ax = axes[0, 0]
    scatter = ax.scatter(v_space[:, 0], v_space[:, 1],
                        c=v_clusters, cmap='tab10', s=1, alpha=0.5)
    ax.set_xlabel('V1', fontweight='bold')
    ax.set_ylabel('V2', fontweight='bold')
    ax.set_title('V-space Clusters', fontweight='bold', fontsize=12)
    plt.colorbar(scatter, ax=ax, label='Cluster')

    # Plot 2: V-space by batch (top batches)
    ax = axes[0, 1]
    if batch_key:
        top_batches = adata.obs[batch_key].value_counts().head(10).index
        batch_colors = {batch: i for i, batch in enumerate(top_batches)}
        colors = [batch_colors.get(b, -1) for b in adata.obs[batch_key]]
        scatter = ax.scatter(v_space[:, 0], v_space[:, 1],
                            c=colors, cmap='tab20', s=1, alpha=0.5)
        ax.set_xlabel('V1', fontweight='bold')
        ax.set_ylabel('V2', fontweight='bold')
        ax.set_title('V-space by Top 10 Batches', fontweight='bold', fontsize=12)

    # Plot 3: V-space by cell type (top types)
    ax = axes[0, 2]
    if cell_type_key:
        top_types = adata.obs[cell_type_key].value_counts().head(10).index
        type_colors = {ct: i for i, ct in enumerate(top_types)}
        colors = [type_colors.get(ct, -1) for ct in adata.obs[cell_type_key]]
        scatter = ax.scatter(v_space[:, 0], v_space[:, 1],
                            c=colors, cmap='tab20', s=1, alpha=0.5)
        ax.set_xlabel('V1', fontweight='bold')
        ax.set_ylabel('V2', fontweight='bold')
        ax.set_title('V-space by Top 10 Cell Types', fontweight='bold', fontsize=12)

    # Plot 4: V-space by pseudotime
    ax = axes[1, 0]
    if pseudotime is not None:
        valid_mask = np.isfinite(pseudotime)
        scatter = ax.scatter(v_space[valid_mask, 0], v_space[valid_mask, 1],
                            c=pseudotime[valid_mask], cmap='viridis', s=1, alpha=0.5)
        ax.set_xlabel('V1', fontweight='bold')
        ax.set_ylabel('V2', fontweight='bold')
        ax.set_title('V-space by Pseudotime', fontweight='bold', fontsize=12)
        plt.colorbar(scatter, ax=ax, label='Pseudotime')

    # Plot 5: Batch composition per cluster
    ax = axes[1, 1]
    if batch_key:
        cluster_batch_counts = pd.crosstab(adata.obs['v_cluster'],
                                           adata.obs[batch_key],
                                           normalize='index') * 100
        # Show top 10 batches
        top_10_batches = adata.obs[batch_key].value_counts().head(10).index
        cluster_batch_counts[top_10_batches].T.plot(kind='bar', ax=ax, stacked=True)
        ax.set_xlabel('Batch', fontweight='bold')
        ax.set_ylabel('% of cells in cluster', fontweight='bold')
        ax.set_title('Batch Composition by V-cluster', fontweight='bold', fontsize=12)
        ax.legend(title='V-cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 6: Cell type composition per cluster
    ax = axes[1, 2]
    if cell_type_key:
        cluster_type_counts = pd.crosstab(adata.obs['v_cluster'],
                                          adata.obs[cell_type_key],
                                          normalize='index') * 100
        # Show top 10 types
        top_10_types = adata.obs[cell_type_key].value_counts().head(10).index
        cluster_type_counts[top_10_types].T.plot(kind='bar', ax=ax, stacked=True)
        ax.set_xlabel('Cell Type', fontweight='bold')
        ax.set_ylabel('% of cells in cluster', fontweight='bold')
        ax.set_title('Cell Type Composition by V-cluster', fontweight='bold', fontsize=12)
        ax.legend(title='V-cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.suptitle(f'V-space Structure Analysis: {exp_name}',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(f'{output_prefix}_v_space_investigation.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization: {output_prefix}_v_space_investigation.png")
    plt.close()

def main():
    """Investigate key models."""

    print("="*80)
    print("V-SPACE STRUCTURE INVESTIGATION")
    print("="*80)

    models = [
        ('Regular Decipher',
         'bonemarrowmap_small_regular_decipher.h5ad',
         'regular_decipher'),
        ('Beta=1.0 + 2 Heads',
         'Beta_1.0_AttentionHeads_2_Training/bonemarrowmap_small_150epochs_beta1_attnheads2.h5ad',
         'beta1_attnheads2'),
        ('Beta=2.0 (4 heads)',
         'Beta_2.0_Training/bonemarrowmap_small_150epochs_beta2.0.h5ad',
         'beta2'),
    ]

    for exp_name, path, prefix in models:
        investigate_v_space(path, exp_name, prefix)

    print("\n" + "="*80)
    print("INVESTIGATION COMPLETE")
    print("="*80)
    print()

if __name__ == "__main__":
    main()
