"""
Compute pseudotime for all batch-corrected models and analyze gradient quality.
"""

import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

def compute_and_analyze_pseudotime(adata_path, exp_name, use_batch_corrected=True):
    """Compute pseudotime and analyze gradient quality."""

    print(f"\nProcessing: {exp_name}")
    print(f"  Loading: {adata_path}")

    adata = sc.read_h5ad(adata_path)

    # Determine which representation to use
    if use_batch_corrected and 'X_decipher_batch_corrected_z' in adata.obsm:
        rep_key = 'X_decipher_batch_corrected_z'
        v_key = 'X_decipher_batch_corrected_v'
        print(f"  Using batch-corrected embeddings")
    elif 'decipher_z' in adata.obsm:
        rep_key = 'decipher_z'
        v_key = 'decipher_v'
        print(f"  Using regular decipher embeddings")
    else:
        print(f"  ✗ No suitable embeddings found")
        return None

    # Find HSC cells for root
    cell_type_col = None
    for col in adata.obs.columns:
        if 'celltype' in col.lower() or 'cell_type' in col.lower():
            cell_type_col = col
            break

    if cell_type_col is None:
        print(f"  ✗ No cell type column found")
        return None

    # Find HSC cells
    hsc_mask = adata.obs[cell_type_col].str.contains('HSC', case=False, na=False)
    if hsc_mask.sum() == 0:
        print(f"  ✗ No HSC cells found")
        return None

    # Set root cell as the first HSC
    root_idx = np.where(hsc_mask)[0][0]
    adata.uns['iroot'] = root_idx

    print(f"  Computing diffusion pseudotime...")
    print(f"    Root: HSC cell at index {root_idx}")

    # Compute neighbors and diffusion pseudotime on latent z
    sc.pp.neighbors(adata, use_rep=rep_key, n_neighbors=15)
    sc.tl.diffmap(adata)
    sc.tl.dpt(adata)

    pseudotime = adata.obs['dpt_pseudotime'].values
    v_space = adata.obsm[v_key]

    # Remove infinite/nan values
    valid_mask = np.isfinite(pseudotime)
    n_valid = valid_mask.sum()
    n_total = len(pseudotime)

    print(f"  Pseudotime computed:")
    print(f"    Valid cells: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")
    print(f"    Range: [{np.nanmin(pseudotime):.3f}, {np.nanmax(pseudotime):.3f}]")

    if n_valid < 100:
        print(f"  ✗ Too few valid pseudotime values")
        return None

    # Compute V-space distance vs pseudotime correlation
    from scipy.spatial.distance import pdist, squareform

    # Sample for speed
    sample_size = min(2000, n_valid)
    valid_indices = np.where(valid_mask)[0]
    sample_idx = np.random.choice(valid_indices, sample_size, replace=False)

    pt_sample = pseudotime[sample_idx]
    v_sample = v_space[sample_idx]

    # Compute pairwise distances
    v_distances = squareform(pdist(v_sample))
    pt_distances = squareform(pdist(pt_sample.reshape(-1, 1)))

    # Flatten upper triangle
    triu_idx = np.triu_indices_from(v_distances, k=1)
    v_flat = v_distances[triu_idx]
    pt_flat = pt_distances[triu_idx]

    # Compute correlation
    spearman_corr, p_value = spearmanr(v_flat, pt_flat)

    print(f"  V-pseudotime correlation (Spearman): {spearman_corr:.4f} (p={p_value:.2e})")

    # Analyze connectivity
    # Check if pseudotime forms connected components
    pt_bins = pd.cut(pseudotime[valid_mask], bins=10)
    bin_counts = pt_bins.value_counts().sort_index()

    print(f"  Pseudotime distribution across 10 bins:")
    for bin_label, count in bin_counts.head(5).items():
        print(f"    {bin_label}: {count} cells")

    # Check for "islands" - disconnected regions
    # Use UMAP to visualize
    if 'X_umap' not in adata.obsm or 'X_decipher_batch_corrected_umap' in adata.obsm:
        print(f"  Computing UMAP for visualization...")
        sc.tl.umap(adata, neighbors_key=None)  # Use the neighbors we just computed

    # Save updated adata with pseudotime
    adata.write(adata_path)
    print(f"  ✓ Saved updated data with pseudotime")

    return {
        'experiment': exp_name,
        'n_cells': n_total,
        'n_valid_pseudotime': n_valid,
        'pct_valid': 100 * n_valid / n_total,
        'pseudotime_min': np.nanmin(pseudotime),
        'pseudotime_max': np.nanmax(pseudotime),
        'v_pt_correlation': spearman_corr,
        'v_pt_pvalue': p_value,
    }

def main():
    """Process all experiments."""

    print("=" * 80)
    print("COMPUTING PSEUDOTIME FOR ALL MODELS")
    print("=" * 80)

    experiments = [
        ('Regular Decipher', 'bonemarrowmap_small_regular_decipher.h5ad', False),
        ('Beta=1.0 + 2 Heads', 'Beta_1.0_AttentionHeads_2_Training/bonemarrowmap_small_150epochs_beta1_attnheads2.h5ad', True),
        ('Beta=0.5 (4 heads)', 'Beta_0.5_Training/bonemarrowmap_small_150epochs_beta0.5.h5ad', True),
        ('Beta=1.0 (4 heads)', 'Beta_1.0_Training/bonemarrowmap_small_150epochs_beta1.h5ad', True),
        ('Beta=2.0 (4 heads)', 'Beta_2.0_Training/bonemarrowmap_small_150epochs_beta2.0.h5ad', True),
        ('Beta=0.1 (2 heads)', 'AttentionHeads_2_Training/bonemarrowmap_small_150epochs_attnheads2.h5ad', True),
        ('Beta=0.1 (4 heads)', 'Full_Training/bonemarrowmap_small_150epochs_batch_corrected.h5ad', True),
    ]

    results = []

    for exp_name, path, use_bc in experiments:
        result = compute_and_analyze_pseudotime(path, exp_name, use_bc)
        if result:
            results.append(result)

    # Create comparison table
    df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("PSEUDOTIME ANALYSIS COMPARISON")
    print("=" * 80)
    print()

    print(f"{'Experiment':<30} {'Valid %':>10} {'PT Range':>15} {'V-PT Corr':>12} {'P-value':>12}")
    print("-" * 85)

    for _, row in df.iterrows():
        exp = row['experiment']
        valid_pct = row['pct_valid']
        pt_range = f"[{row['pseudotime_min']:.2f}, {row['pseudotime_max']:.2f}]"
        corr = row['v_pt_correlation']
        pval = row['v_pt_pvalue']

        print(f"{exp:<30} {valid_pct:>10.1f} {pt_range:>15} {corr:>12.4f} {pval:>12.2e}")

    print("\n" + "=" * 80)
    print("KEY OBSERVATIONS")
    print("=" * 80)
    print()

    # Best correlation
    best_idx = df['v_pt_correlation'].idxmax()
    best = df.iloc[best_idx]

    print(f"✓ Best V-Pseudotime Correlation:")
    print(f"  {best['experiment']}: {best['v_pt_correlation']:.4f}")
    print()

    # Check for problems
    print("⚠ Potential Issues:")
    for _, row in df.iterrows():
        if row['pct_valid'] < 100:
            print(f"  • {row['experiment']}: Only {row['pct_valid']:.1f}% cells have valid pseudotime")
        if row['v_pt_correlation'] < 0.4:
            print(f"  • {row['experiment']}: Weak V-pseudotime correlation ({row['v_pt_correlation']:.4f})")

    print()

    # Save results
    df.to_csv('pseudotime_analysis_results.csv', index=False)
    print("✓ Saved results to: pseudotime_analysis_results.csv")
    print()

    print("=" * 80)

if __name__ == "__main__":
    np.random.seed(42)
    main()
