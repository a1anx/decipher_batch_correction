"""
Comprehensive batch mixing and pseudotime analysis for all experiments.

This script computes:
1. Batch mixing metrics (silhouette scores for UMAP and V-space)
2. Pseudotime gradient quality
3. Batch-specific statistics
"""

import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

def compute_batch_mixing_metrics(adata, batch_key='Donor'):
    """Compute comprehensive batch mixing metrics."""

    metrics = {}

    # UMAP batch mixing
    if 'X_umap' in adata.obsm:
        silh_umap = silhouette_score(adata.obsm['X_umap'], adata.obs[batch_key])
        metrics['UMAP_batch_silhouette'] = silh_umap

    # V-space batch mixing
    if 'X_decipher_batch_corrected_v' in adata.obsm:
        silh_v = silhouette_score(adata.obsm['X_decipher_batch_corrected_v'], adata.obs[batch_key])
        metrics['V_space_batch_silhouette'] = silh_v
    elif 'decipher_v' in adata.obsm:
        silh_v = silhouette_score(adata.obsm['decipher_v'], adata.obs[batch_key])
        metrics['V_space_batch_silhouette'] = silh_v

    # Batch entropy (measure of mixing)
    # For each cell, compute entropy of its k-nearest neighbors' batches
    if 'X_umap' in adata.obsm:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=30)
        nn.fit(adata.obsm['X_umap'])
        distances, indices = nn.kneighbors(adata.obsm['X_umap'])

        batch_entropy = []
        for i in range(len(adata)):
            neighbor_batches = adata.obs[batch_key].iloc[indices[i]].value_counts(normalize=True)
            entropy = -np.sum(neighbor_batches * np.log(neighbor_batches + 1e-10))
            batch_entropy.append(entropy)

        metrics['mean_batch_entropy'] = np.mean(batch_entropy)
        metrics['std_batch_entropy'] = np.std(batch_entropy)

    return metrics

def analyze_pseudotime_gradient(adata):
    """Analyze quality of pseudotime gradient in V-space."""

    if 'dpt_pseudotime' not in adata.obs:
        return None

    if 'X_decipher_batch_corrected_v' in adata.obsm:
        v_space = adata.obsm['X_decipher_batch_corrected_v']
    elif 'decipher_v' in adata.obsm:
        v_space = adata.obsm['decipher_v']
    else:
        return None

    pseudotime = adata.obs['dpt_pseudotime'].values

    # Remove infinite values
    valid_mask = np.isfinite(pseudotime)
    pseudotime = pseudotime[valid_mask]
    v_space = v_space[valid_mask]

    # Compute correlation between V-space distance and pseudotime
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import spearmanr, pearsonr

    # Sample for faster computation
    if len(pseudotime) > 2000:
        sample_idx = np.random.choice(len(pseudotime), 2000, replace=False)
        pseudotime_sample = pseudotime[sample_idx]
        v_space_sample = v_space[sample_idx]
    else:
        pseudotime_sample = pseudotime
        v_space_sample = v_space

    # Compute pairwise distances
    v_distances = squareform(pdist(v_space_sample))
    pseudotime_distances = squareform(pdist(pseudotime_sample.reshape(-1, 1)))

    # Flatten for correlation
    v_flat = v_distances[np.triu_indices_from(v_distances, k=1)]
    pt_flat = pseudotime_distances[np.triu_indices_from(pseudotime_distances, k=1)]

    # Compute correlations
    spearman_corr, _ = spearmanr(v_flat, pt_flat)
    pearson_corr, _ = pearsonr(v_flat, pt_flat)

    return {
        'spearman_v_pseudotime_corr': spearman_corr,
        'pearson_v_pseudotime_corr': pearson_corr,
        'pseudotime_range': (np.min(pseudotime), np.max(pseudotime)),
        'n_cells_with_pseudotime': len(pseudotime)
    }

def main():
    """Analyze all experiments."""

    print("=" * 80)
    print("COMPREHENSIVE BATCH MIXING AND PSEUDOTIME ANALYSIS")
    print("=" * 80)
    print()

    experiments = {
        'Regular Decipher': {
            'file': 'bonemarrowmap_small_regular_decipher.h5ad',
            'has_batch_correction': False
        },
        'Beta=1.0 + 2 Heads': {
            'file': 'Beta_1.0_AttentionHeads_2_Training/bonemarrowmap_small_150epochs_beta1_attnheads2.h5ad',
            'has_batch_correction': True
        },
        'Beta=0.5 (4 heads)': {
            'file': 'Beta_0.5_Training/bonemarrowmap_small_150epochs_beta0.5.h5ad',
            'has_batch_correction': True
        },
        'Beta=1.0 (4 heads)': {
            'file': 'Beta_1.0_Training/bonemarrowmap_small_150epochs_beta1.h5ad',
            'has_batch_correction': True
        },
        'Beta=2.0 (4 heads)': {
            'file': 'Beta_2.0_Training/bonemarrowmap_small_150epochs_beta2.0.h5ad',
            'has_batch_correction': True
        },
        'Beta=0.1 (2 heads)': {
            'file': 'AttentionHeads_2_Training/bonemarrowmap_small_150epochs_attnheads2.h5ad',
            'has_batch_correction': True
        },
        'Beta=0.1 (4 heads)': {
            'file': 'Full_Training/bonemarrowmap_small_150epochs_batch_corrected.h5ad',
            'has_batch_correction': True
        },
    }

    results = []

    for exp_name, exp_info in experiments.items():
        try:
            print(f"\nAnalyzing: {exp_name}")
            print(f"  Loading {exp_info['file']}...")

            adata = sc.read_h5ad(exp_info['file'])

            # Get batch key
            batch_key = None
            for col in adata.obs.columns:
                if 'donor' in col.lower() or 'batch' in col.lower():
                    batch_key = col
                    break

            if batch_key is None:
                print(f"  ✗ No batch column found")
                continue

            # Compute metrics
            mixing_metrics = compute_batch_mixing_metrics(adata, batch_key)
            pseudotime_metrics = analyze_pseudotime_gradient(adata)

            result = {
                'Experiment': exp_name,
                **mixing_metrics
            }

            if pseudotime_metrics:
                result.update(pseudotime_metrics)

            results.append(result)

            print(f"  ✓ Metrics computed:")
            print(f"    V-space batch silhouette: {mixing_metrics.get('V_space_batch_silhouette', 'N/A'):.4f}")
            print(f"    UMAP batch silhouette: {mixing_metrics.get('UMAP_batch_silhouette', 'N/A'):.4f}")
            print(f"    Mean batch entropy: {mixing_metrics.get('mean_batch_entropy', 'N/A'):.4f}")
            if pseudotime_metrics:
                print(f"    V-pseudotime correlation (Spearman): {pseudotime_metrics['spearman_v_pseudotime_corr']:.4f}")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue

    # Create DataFrame
    df = pd.DataFrame(results)

    # Print comparison table
    print("\n" + "=" * 80)
    print("BATCH MIXING COMPARISON TABLE")
    print("=" * 80)
    print()

    print(f"{'Experiment':<30} {'V_silh':>10} {'UMAP_silh':>10} {'Entropy':>10} {'V-PT_corr':>12}")
    print("-" * 75)

    for _, row in df.iterrows():
        exp = row['Experiment']
        v_silh = row.get('V_space_batch_silhouette', np.nan)
        umap_silh = row.get('UMAP_batch_silhouette', np.nan)
        entropy = row.get('mean_batch_entropy', np.nan)
        v_pt_corr = row.get('spearman_v_pseudotime_corr', np.nan)

        print(f"{exp:<30} {v_silh:>10.4f} {umap_silh:>10.4f} {entropy:>10.4f} {v_pt_corr:>12.4f}")

    print("\n" + "=" * 80)
    print("METRIC INTERPRETATION")
    print("=" * 80)
    print()
    print("Batch Silhouette Score:")
    print("  • Range: [-1, 1]")
    print("  • More NEGATIVE = better batch mixing")
    print("  • More POSITIVE = batches are separated (BAD for batch correction)")
    print("  • Close to 0 = batches are well-mixed")
    print()
    print("Batch Entropy:")
    print("  • Range: [0, log(n_batches)]")
    print("  • Higher = better mixing (neighbors from diverse batches)")
    print("  • Max = log(45) ≈ 3.81 for perfect mixing")
    print()
    print("V-Pseudotime Correlation (Spearman):")
    print("  • Range: [-1, 1]")
    print("  • Higher = V-space preserves pseudotime gradient better")
    print("  • >0.5 = good preservation of trajectory structure")
    print()

    # Identify best performers
    print("=" * 80)
    print("BEST PERFORMERS")
    print("=" * 80)
    print()

    # Best batch mixing (most negative V-space silhouette)
    best_mixing_idx = df['V_space_batch_silhouette'].idxmin()
    best_mixing = df.iloc[best_mixing_idx]
    print(f"Best Batch Mixing (V-space):")
    print(f"  {best_mixing['Experiment']}")
    print(f"  V-space silhouette: {best_mixing['V_space_batch_silhouette']:.4f}")
    print()

    # Best pseudotime preservation
    if 'spearman_v_pseudotime_corr' in df.columns:
        best_pt_idx = df['spearman_v_pseudotime_corr'].idxmax()
        best_pt = df.iloc[best_pt_idx]
        print(f"Best Pseudotime Gradient Preservation:")
        print(f"  {best_pt['Experiment']}")
        print(f"  V-pseudotime correlation: {best_pt['spearman_v_pseudotime_corr']:.4f}")
        print()

    # Best overall balance
    # Normalize metrics and compute composite score
    df_normalized = df.copy()

    # For silhouette: want more negative, so flip sign
    df_normalized['V_batch_mixing_score'] = -df_normalized['V_space_batch_silhouette']
    df_normalized['UMAP_batch_mixing_score'] = -df_normalized['UMAP_batch_silhouette']

    # Normalize to [0, 1]
    for col in ['V_batch_mixing_score', 'UMAP_batch_mixing_score', 'mean_batch_entropy', 'spearman_v_pseudotime_corr']:
        if col in df_normalized.columns:
            min_val = df_normalized[col].min()
            max_val = df_normalized[col].max()
            if max_val > min_val:
                df_normalized[col + '_norm'] = (df_normalized[col] - min_val) / (max_val - min_val)
            else:
                df_normalized[col + '_norm'] = 0.5

    # Composite score: weight all metrics equally
    df_normalized['composite_score'] = (
        df_normalized.get('V_batch_mixing_score_norm', 0) * 0.3 +
        df_normalized.get('UMAP_batch_mixing_score_norm', 0) * 0.2 +
        df_normalized.get('mean_batch_entropy_norm', 0) * 0.2 +
        df_normalized.get('spearman_v_pseudotime_corr_norm', 0) * 0.3
    )

    best_overall_idx = df_normalized['composite_score'].idxmax()
    best_overall = df.iloc[best_overall_idx]

    print(f"Best Overall Balance:")
    print(f"  {best_overall['Experiment']}")
    print(f"  Composite score: {df_normalized.iloc[best_overall_idx]['composite_score']:.4f}")
    print(f"  V-space silhouette: {best_overall['V_space_batch_silhouette']:.4f}")
    print(f"  Batch entropy: {best_overall['mean_batch_entropy']:.4f}")
    print(f"  V-pseudotime corr: {best_overall.get('spearman_v_pseudotime_corr', np.nan):.4f}")
    print()

    # Save results
    df.to_csv('batch_mixing_analysis_results.csv', index=False)
    print("✓ Saved detailed results to: batch_mixing_analysis_results.csv")
    print()

    # Create visualization
    create_mixing_visualization(df)

    print("=" * 80)

def create_mixing_visualization(df):
    """Create visualization of batch mixing metrics."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Sort by V-space silhouette
    df_sorted = df.sort_values('V_space_batch_silhouette')

    # Plot 1: V-space batch silhouette
    ax1 = axes[0, 0]
    bars = ax1.barh(df_sorted['Experiment'], df_sorted['V_space_batch_silhouette'],
                     color=['green' if x < -0.27 else 'orange' if x < -0.25 else 'red'
                           for x in df_sorted['V_space_batch_silhouette']])
    ax1.set_xlabel('V-space Batch Silhouette\n(more negative = better mixing)', fontweight='bold')
    ax1.set_title('V-space Batch Mixing', fontweight='bold')
    ax1.axvline(-0.2840, color='blue', linestyle='--', label='Regular Decipher', alpha=0.7)
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        val = row['V_space_batch_silhouette']
        ax1.text(val - 0.01, i, f'{val:.4f}', ha='right', va='center', fontsize=8, fontweight='bold')

    # Plot 2: UMAP batch silhouette
    ax2 = axes[0, 1]
    df_sorted2 = df.sort_values('UMAP_batch_silhouette')
    bars = ax2.barh(df_sorted2['Experiment'], df_sorted2['UMAP_batch_silhouette'],
                     color=['green' if x < -0.26 else 'orange' if x < -0.24 else 'red'
                           for x in df_sorted2['UMAP_batch_silhouette']])
    ax2.set_xlabel('UMAP Batch Silhouette\n(more negative = better mixing)', fontweight='bold')
    ax2.set_title('UMAP Batch Mixing', fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (idx, row) in enumerate(df_sorted2.iterrows()):
        val = row['UMAP_batch_silhouette']
        ax2.text(val - 0.005, i, f'{val:.4f}', ha='right', va='center', fontsize=8, fontweight='bold')

    # Plot 3: Batch entropy
    ax3 = axes[1, 0]
    df_sorted3 = df.sort_values('mean_batch_entropy', ascending=False)
    bars = ax3.barh(df_sorted3['Experiment'], df_sorted3['mean_batch_entropy'],
                     color=['green' if x > 2.5 else 'orange' if x > 2.3 else 'red'
                           for x in df_sorted3['mean_batch_entropy']])
    ax3.set_xlabel('Mean Batch Entropy\n(higher = better mixing)', fontweight='bold')
    ax3.set_title('Batch Diversity in Neighborhoods', fontweight='bold')
    ax3.axvline(np.log(45), color='blue', linestyle='--', label='Perfect mixing', alpha=0.7)
    ax3.legend()
    ax3.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (idx, row) in enumerate(df_sorted3.iterrows()):
        val = row['mean_batch_entropy']
        ax3.text(val + 0.05, i, f'{val:.3f}', ha='left', va='center', fontsize=8, fontweight='bold')

    # Plot 4: V-pseudotime correlation
    ax4 = axes[1, 1]
    if 'spearman_v_pseudotime_corr' in df.columns:
        df_sorted4 = df.sort_values('spearman_v_pseudotime_corr', ascending=False)
        bars = ax4.barh(df_sorted4['Experiment'], df_sorted4['spearman_v_pseudotime_corr'],
                         color=['green' if x > 0.5 else 'orange' if x > 0.4 else 'red'
                               for x in df_sorted4['spearman_v_pseudotime_corr']])
        ax4.set_xlabel('V-Pseudotime Correlation (Spearman)\n(higher = better trajectory)', fontweight='bold')
        ax4.set_title('Pseudotime Gradient Preservation', fontweight='bold')
        ax4.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, (idx, row) in enumerate(df_sorted4.iterrows()):
            val = row['spearman_v_pseudotime_corr']
            ax4.text(val + 0.01, i, f'{val:.3f}', ha='left', va='center', fontsize=8, fontweight='bold')

    plt.suptitle('Comprehensive Batch Mixing and Trajectory Quality Analysis',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig('batch_mixing_analysis_visualization.png', dpi=300, bbox_inches='tight')
    print("✓ Saved visualization to: batch_mixing_analysis_visualization.png")
    plt.close()

if __name__ == "__main__":
    main()
