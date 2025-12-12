"""
Delta Titration Curve Analysis - Scaled Magnitude Approach

This script implements a systematic titration experiment to determine the critical
threshold where batch correction fails.

Approach:
- Fixed structure: Always 4 batches with pattern [0, 1, 2, 3]
- Scaled magnitude: Multiply by different factors (0.01 to 5.0)
- Delta=0 baseline: Present in all datasets as reference
- Single parameter: Magnitude controls batch effect strength

Results:
- Titration curves showing batch correction performance vs magnitude
- Critical threshold identification
- Per-batch and overall mixing metrics
"""

import sys
from pathlib import Path
import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.stats import spearmanr

# Add parent directory to import decipher batch correction modules
parent_dir = Path(__file__).parent.parent / "decipher-batch-correction" / "decipher-bc"
sys.path.insert(0, str(parent_dir))

from decipher_batch_corrected import DecipherBatchCorrectedConfig
from train_batch_corrected import train_batch_corrected_decipher, evaluate_batch_correction

# Import from simulation script
from simulations1209 import simulation_correlated_shift, run_methods

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_LOGGER = logging.getLogger(__name__)


def generate_scaled_delta_scenarios(
    base_structure=[0, 1, 2, 3],
    magnitudes=None
):
    """
    Generate delta scenarios by scaling a base structure.
    Delta=0 is always included as a baseline batch.

    Parameters:
    -----------
    base_structure : list
        Base increment pattern starting from 0 (default: [0, 1, 2, 3])
    magnitudes : list
        Scaling factors to apply (default: [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0])

    Returns:
    --------
    dict : {magnitude: {'deltas': [...], 'increment': ..., ...}}
    """
    if magnitudes is None:
        # Strategic magnitudes from very small to extreme
        magnitudes = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0]

    scenarios = {}
    for mag in magnitudes:
        deltas = [x * mag for x in base_structure]

        scenarios[mag] = {
            'deltas': deltas,
            'magnitude': mag,
            'increment': mag,  # Increment between consecutive batches
            'n_batches': len(deltas),
            'max_delta': max(deltas),
            'min_delta': 0.0,  # Always starts at 0
            'range': max(deltas),
            'description': f"{len(deltas)} batches: {deltas}"
        }

    return scenarios


def compute_batch_mixing_metrics(adata, batch_key='delta', z_key='X_decipher_batch_corrected_z'):
    """
    Compute comprehensive metrics to assess batch mixing quality.

    Metrics:
    - batch_silhouette: Lower = better batch mixing
    - pseudotime_correlation: Higher = better biological preservation
    - cluster_ari: Higher = better biological structure preservation
    - mean_attention_strength: Model's batch correction effort

    Returns:
        dict: Dictionary of metric values
    """
    metrics = {}

    # 1. Batch Silhouette Score (lower = better mixing)
    try:
        batch_silhouette = silhouette_score(
            adata.obsm[z_key],
            adata.obs[batch_key]
        )
        metrics['batch_silhouette'] = batch_silhouette
    except Exception as e:
        _LOGGER.warning(f"Could not compute batch silhouette: {e}")
        metrics['batch_silhouette'] = np.nan

    # 2. Biological Signal Preservation (Spearman correlation with pseudotime)
    if 'latent_t' in adata.obs.columns:
        try:
            # Project to 1D using first principal component
            z_embedding = adata.obsm[z_key]
            pc1 = z_embedding @ np.linalg.svd(z_embedding, full_matrices=False)[2][0]

            corr, pval = spearmanr(pc1, adata.obs['latent_t'])
            metrics['pseudotime_correlation'] = abs(corr)
            metrics['pseudotime_pval'] = pval
        except Exception as e:
            _LOGGER.warning(f"Could not compute pseudotime correlation: {e}")
            metrics['pseudotime_correlation'] = np.nan
            metrics['pseudotime_pval'] = np.nan

    # 3. Cluster ARI (biological structure preservation)
    if 'cluster_true' in adata.obs.columns:
        try:
            from sklearn.cluster import KMeans
            n_clusters = len(adata.obs['cluster_true'].unique())
            kmeans = KMeans(n_clusters=n_clusters, random_state=0)
            pred_clusters = kmeans.fit_predict(adata.obsm[z_key])

            ari = adjusted_rand_score(adata.obs['cluster_true'], pred_clusters)
            metrics['cluster_ari'] = ari
        except Exception as e:
            _LOGGER.warning(f"Could not compute cluster ARI: {e}")
            metrics['cluster_ari'] = np.nan

    # 4. Mean attention strength (if available)
    if 'batch_attention_strength' in adata.obs.columns:
        metrics['mean_attention_strength'] = adata.obs['batch_attention_strength'].mean()

    return metrics


def run_scaled_magnitude_titration(
    base_structure=[0, 1, 2, 3],
    magnitudes=None,
    n_samples_per_batch=250,
    n_genes=50,
    sigma=0.05,
    branch_prob=0.7,
    k_clusters=20,
    hole_size=1,
    n_holes=3,
    hole_density=0.05,
    seed=0,
    out_folder="delta_titration_results",
    adata_folder="delta_titration_adata",
    skip_training=False,
):
    """
    Run delta titration experiment with scaled magnitude approach.

    Each dataset contains 4 batches: [0*mag, 1*mag, 2*mag, 3*mag]
    where mag increases from 0.01 to 5.0.

    Parameters:
    -----------
    base_structure : list
        Base pattern to scale (default: [0, 1, 2, 3])
    magnitudes : list
        Scaling factors (default: [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0])
    n_samples_per_batch : int
        Number of cells per batch
    skip_training : bool
        If True, skip batch correction training (for testing)

    Returns:
    --------
    pd.DataFrame : Results with metrics for each magnitude
    """
    if magnitudes is None:
        magnitudes = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0]

    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(adata_folder, exist_ok=True)

    results = []

    print("=" * 70)
    print("DELTA TITRATION CURVE - SCALED MAGNITUDE APPROACH")
    print("=" * 70)
    print(f"\nBase structure: {base_structure}")
    print(f"Testing {len(magnitudes)} magnitude scales")
    print(f"Samples per batch: {n_samples_per_batch}")
    print(f"Total cells per dataset: {n_samples_per_batch * len(base_structure)}")
    print("")

    for i, mag in enumerate(magnitudes):
        deltas = [x * mag for x in base_structure]

        print("=" * 70)
        print(f"[{i+1}/{len(magnitudes)}] Magnitude = {mag:.2f}")
        print(f"  Deltas: {[f'{d:.2f}' for d in deltas]}")
        print(f"  Increment: {mag:.2f}")
        print(f"  Range: 0.00 - {max(deltas):.2f}")
        print("=" * 70)

        # Generate concatenated data with multiple delta batches
        print(f"\n[Step 1/5] Generating simulated data...")
        adata_list = []
        for delta in deltas:
            adata_single = simulation_correlated_shift(
                n_samples=n_samples_per_batch,
                n_genes=n_genes,
                seed=seed,
                sigma=sigma,
                branch_prob=branch_prob,
                k_clusters=k_clusters,
                hole_size=hole_size,
                n_holes=n_holes,
                hole_density=hole_density,
                delta=delta,
            )
            adata_list.append(adata_single)

        # Concatenate all batches
        adata_combined = sc.concat(adata_list, label='delta', keys=[str(d) for d in deltas])
        print(f"✓ Generated {adata_combined.shape[0]} cells across {len(deltas)} batches")

        # Run baseline methods (original Decipher, UMAP, etc.)
        print(f"\n[Step 2/5] Running baseline methods...")
        latent_spaces = run_methods(adata_combined, seed=seed)
        print(f"✓ Computed {len(latent_spaces)} baseline embeddings")

        # Compute metrics on original Decipher
        print(f"\n[Step 3/5] Computing metrics on original Decipher...")
        original_metrics = compute_batch_mixing_metrics(
            adata_combined,
            batch_key='delta',
            z_key='decipher_decipher_z'
        )
        original_metrics['magnitude'] = mag
        original_metrics['max_delta'] = max(deltas)
        original_metrics['increment'] = mag
        original_metrics['method'] = 'original'
        results.append(original_metrics)
        print(f"✓ Original Silhouette: {original_metrics['batch_silhouette']:.4f}")

        if not skip_training:
            # Train batch-corrected Decipher
            print(f"\n[Step 4/5] Training batch-corrected Decipher...")
            try:
                config = DecipherBatchCorrectedConfig(
                    dim_z=10,
                    dim_v=2,
                    n_batches=None,
                    batch_emb_dim=32,
                    decoder_hidden_dims=[64, 128],
                    n_attention_heads=4,
                    combination_mode="concat",
                    learning_rate=5e-3,
                    batch_size=128,
                    n_epochs=100,
                    early_stopping_patience=15
                )

                config.initialize_from_adata(adata_combined, batch_key='delta')

                model, losses = train_batch_corrected_decipher(
                    adata_combined,
                    batch_key='delta',
                    config=config,
                    device='cpu'
                )

                # Extract batch-corrected embeddings
                bc_results = evaluate_batch_correction(model, adata_combined, batch_key='delta', device='cpu')
                adata_combined.obsm['X_decipher_batch_corrected_z'] = bc_results['z']
                adata_combined.obsm['X_decipher_batch_corrected_v'] = bc_results['v']
                adata_combined.obs['batch_attention_strength'] = bc_results['attention_weights'].mean(axis=(1, 2, 3))

                print(f"✓ Training complete (final loss: {losses['val_losses'][-1]:.2f})")

                # Compute metrics on batch-corrected embeddings
                print(f"\n[Step 5/5] Computing batch-corrected metrics...")
                bc_metrics = compute_batch_mixing_metrics(
                    adata_combined,
                    batch_key='delta',
                    z_key='X_decipher_batch_corrected_z'
                )
                bc_metrics['magnitude'] = mag
                bc_metrics['max_delta'] = max(deltas)
                bc_metrics['increment'] = mag
                bc_metrics['method'] = 'batch_corrected'
                results.append(bc_metrics)
                print(f"✓ Batch-corrected Silhouette: {bc_metrics['batch_silhouette']:.4f}")
                print(f"  Improvement: {original_metrics['batch_silhouette'] - bc_metrics['batch_silhouette']:.4f}")

            except Exception as e:
                _LOGGER.error(f"Failed to train batch correction for magnitude={mag}: {e}")
                # Add NaN metrics for this magnitude
                bc_metrics = {
                    'magnitude': mag,
                    'max_delta': max(deltas),
                    'increment': mag,
                    'method': 'batch_corrected',
                    'batch_silhouette': np.nan,
                    'pseudotime_correlation': np.nan,
                    'cluster_ari': np.nan,
                }
                results.append(bc_metrics)

        # Save AnnData
        output_path = f"{adata_folder}/adata_magnitude_{mag:.2f}.h5ad"
        adata_combined.write(output_path)
        print(f"\n✓ Saved: {output_path}\n")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{out_folder}/titration_metrics.csv", index=False)
    print(f"\n✓ Saved metrics: {out_folder}/titration_metrics.csv")

    return results_df


def plot_titration_curves(results_df, out_folder="delta_titration_results"):
    """
    Plot titration curves showing batch correction performance vs magnitude.
    """
    os.makedirs(out_folder, exist_ok=True)
    sns.set_style("whitegrid")

    # Create comprehensive figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics_to_plot = [
        ('batch_silhouette', 'Batch Silhouette Score', 'Lower = Better Mixing'),
        ('pseudotime_correlation', 'Pseudotime Correlation', 'Higher = Better Biological Preservation'),
        ('cluster_ari', 'Cluster ARI', 'Higher = Better Structure Preservation'),
        ('mean_attention_strength', 'Mean Attention Strength', 'Model Batch Correction Effort'),
    ]

    for ax, (metric, title, subtitle) in zip(axes.flat, metrics_to_plot):
        # Plot both original and batch-corrected
        for method in ['original', 'batch_corrected']:
            data = results_df[results_df['method'] == method].sort_values('magnitude')
            if data.empty:
                continue

            label = 'Original Decipher' if method == 'original' else 'Batch-Corrected Decipher'
            marker = 'o' if method == 'original' else 's'
            color = '#d62728' if method == 'original' else '#2ca02c'

            ax.plot(data['magnitude'], data[metric], marker=marker, label=label,
                   linewidth=2, markersize=8, alpha=0.8, color=color)

        ax.set_xlabel('Magnitude (Batch Effect Scale)', fontsize=11, fontweight='bold')
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f'{title}\n{subtitle}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(f"{out_folder}/titration_curves_all_metrics.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {out_folder}/titration_curves_all_metrics.png")
    plt.close()

    # Create focused plot on silhouette score (main metric)
    fig, ax = plt.subplots(figsize=(10, 6))

    for method in ['original', 'batch_corrected']:
        data = results_df[results_df['method'] == method].sort_values('magnitude')
        if data.empty:
            continue

        label = 'Original Decipher' if method == 'original' else 'Batch-Corrected Decipher'
        marker = 'o' if method == 'original' else 's'
        color = '#d62728' if method == 'original' else '#2ca02c'

        ax.plot(data['magnitude'], data['batch_silhouette'], marker=marker, label=label,
               linewidth=3, markersize=10, alpha=0.8, color=color)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Perfect Mixing')
    ax.set_xlabel('Magnitude (Batch Effect Scale)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Batch Silhouette Score', fontsize=13, fontweight='bold')
    ax.set_title('Batch Correction Performance vs Magnitude\n(Lower = Better Batch Mixing)',
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(f"{out_folder}/silhouette_titration.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {out_folder}/silhouette_titration.png")
    plt.close()

    # Calculate and plot improvement
    original_data = results_df[results_df['method'] == 'original'].set_index('magnitude')
    bc_data = results_df[results_df['method'] == 'batch_corrected'].set_index('magnitude')

    if not original_data.empty and not bc_data.empty:
        improvement = original_data['batch_silhouette'] - bc_data['batch_silhouette']

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(improvement.index, improvement.values, marker='D', linewidth=3,
               markersize=10, color='#1f77b4', alpha=0.8)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='No Improvement')
        ax.set_xlabel('Magnitude (Batch Effect Scale)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Silhouette Improvement\n(Original - Batch Corrected)', fontsize=13, fontweight='bold')
        ax.set_title('Batch Correction Improvement vs Magnitude\n(Higher = Better Batch Correction)',
                    fontsize=14, fontweight='bold', pad=15)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

        # Shade region where improvement is positive
        ax.fill_between(improvement.index, 0, improvement.values,
                        where=(improvement.values > 0), alpha=0.2, color='green',
                        label='Effective Batch Correction')

        plt.tight_layout()
        plt.savefig(f"{out_folder}/improvement_titration.png", dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {out_folder}/improvement_titration.png")
        plt.close()


def analyze_critical_threshold(results_df, out_folder="delta_titration_results"):
    """
    Identify the critical magnitude threshold where batch correction starts to fail.
    """
    print("\n" + "=" * 70)
    print("CRITICAL THRESHOLD ANALYSIS")
    print("=" * 70)

    original_data = results_df[results_df['method'] == 'original'].set_index('magnitude')
    bc_data = results_df[results_df['method'] == 'batch_corrected'].set_index('magnitude')

    if original_data.empty or bc_data.empty:
        print("\n⚠ Insufficient data for threshold analysis")
        return {}

    improvement = original_data['batch_silhouette'] - bc_data['batch_silhouette']

    # Find where improvement becomes negative or very small
    effective_magnitudes = improvement[improvement > 0.01].index.tolist()

    if len(effective_magnitudes) > 0:
        max_effective_magnitude = max(effective_magnitudes)
        print(f"\n✓ Batch correction is effective up to magnitude ≈ {max_effective_magnitude:.2f}")
        print(f"  (Improvement > 0.01 in silhouette score)")
    else:
        print("\n⚠ Batch correction shows minimal improvement across all magnitudes")
        max_effective_magnitude = None

    # Find magnitude with maximum improvement
    max_improvement_magnitude = improvement.idxmax()
    max_improvement_value = improvement.max()
    print(f"\n✓ Maximum improvement at magnitude = {max_improvement_magnitude:.2f}")
    print(f"  Improvement: {max_improvement_value:.4f}")

    # Summary table
    print(f"\nSilhouette scores by magnitude:")
    print(f"  Magnitude | Original | Batch-Corrected | Improvement")
    print(f"  " + "-" * 60)
    for mag in sorted(improvement.index):
        orig = original_data.loc[mag, 'batch_silhouette']
        bc = bc_data.loc[mag, 'batch_silhouette']
        imp = improvement.loc[mag]
        print(f"  {mag:9.2f} | {orig:8.4f} | {bc:15.4f} | {imp:11.4f}")

    print("\n" + "=" * 70)

    return {
        'max_effective_magnitude': max_effective_magnitude,
        'max_improvement_magnitude': max_improvement_magnitude,
        'max_improvement_value': max_improvement_value,
    }


if __name__ == "__main__":
    # Define magnitudes to test
    magnitudes = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0]

    # Run titration experiment
    results_df = run_scaled_magnitude_titration(
        base_structure=[0, 1, 2, 3],
        magnitudes=magnitudes,
        n_samples_per_batch=250,
        n_genes=50,
        sigma=0.05,
        branch_prob=0.7,
        k_clusters=20,
        hole_size=1,
        n_holes=3,
        hole_density=0.05,
        seed=0,
        out_folder="delta_titration_results",
        adata_folder="delta_titration_adata",
        skip_training=False,
    )

    # Plot results
    plot_titration_curves(results_df, out_folder="delta_titration_results")

    # Analyze critical threshold
    threshold_analysis = analyze_critical_threshold(results_df, out_folder="delta_titration_results")

    print("\n" + "=" * 70)
    print("✓ DELTA TITRATION ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved in: delta_titration_results/")
    print(f"  - titration_metrics.csv : All metric values")
    print(f"  - titration_curves_all_metrics.png : Multi-metric comparison")
    print(f"  - silhouette_titration.png : Main batch mixing metric")
    print(f"  - improvement_titration.png : Batch correction improvement")
    print(f"\nAnnData files saved in: delta_titration_adata/")
