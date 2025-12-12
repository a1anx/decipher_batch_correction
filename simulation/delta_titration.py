import logging
import pandas as pd
import scanpy as sc
import os
from sklearn.cluster import KMeans
from pathlib import Path
import sys

from matplotlib import pyplot as plt
import seaborn as sns

import matplotlib as mpl

import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
import scanpy as sc
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.stats import spearmanr

import anndata as ad
from simulations1209 import run_delta_shift, simulation_correlated_shift, run_methods, combined_embeddings
import params
from params import SIMUL_PARAMS



parent_dir = Path(__file__).parent.parent / "decipher-batch-correction" / "decipher-bc"
sys.path.insert(0, str(parent_dir))
from decipher_batch_corrected import DecipherBatchCorrectedConfig
from train_batch_corrected import train_batch_corrected_decipher, evaluate_batch_correction


_LOGGER = logging.getLogger(__name__)


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


def delta_magnitudes(
    out_folder,
    adata_folder,
    output_name,
    deltas,
    mag
):
    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(adata_folder, exist_ok=True)
    delta_vec = mag * deltas
    
    # Unshifted Base (delta=0)
    adata_base = simulation_correlated_shift(
        **SIMUL_PARAMS,
        delta=0,
    )
    ground_truth_clusters = adata_base.obs["cluster_true"]
    ground_truth_cluster_rank = adata_base.uns["cluster_rank"] 
    
    # Run all methods on the dataset
    latent_spaces = run_methods(adata_base, seed=params.SEED)
    # _LOGGER.info(f"Latent spaces: {latent_spaces}")
    # adata_base.write(f"{adata_folder}/adata_base.h5ad")
    # _LOGGER.info(f"Adata saved: {adata_folder}/adata_base.h5ad")
    
    #combined_embeddings(adata_base, out_folder, delta=0)
    
    adata_concat = adata_base.copy()

    # Shifted adatas
    for delta in delta_vec:
        adata_sim = simulation_correlated_shift(
            **SIMUL_PARAMS,
            delta=delta,
        )
        # override the clusters
        adata_sim.obs["cluster_true"] = ground_truth_clusters
        adata_sim.uns["cluster_rank"] = ground_truth_cluster_rank
        
        # 4Run all methods on the dataset
        latent_spaces = run_methods(adata_sim, seed=params.SEED)
        
        adata_concat = ad.concat([adata_concat, adata_sim], axis=0,
                                    join="outer", label=None, merge="same")

        # _LOGGER.info(f"Latent spaces: {latent_spaces}")
        # adata_sim.write(f"{adata_folder}/adata_shift_delta_{delta}.h5ad")
        # _LOGGER.info(f"Adata saved: {adata_folder}/adata_shift_delta_{delta}.h5ad")

        # (Optional) keep only 2D spaces if you want pure visual spaces
        # latent_spaces_2d = [k for k in latent_spaces if adata_sim.obsm[k].shape[1] == 2]
        
        # combined_embeddings(adata_sim=adata_sim, out_folder=out_folder, delta=delta)

    adata_concat.write(os.path.join(adata_folder, f"adata_combined_{output_name}.h5ad"))
    _LOGGER.info(f"Combined adata saved: {adata_folder}/adata_combined_{output_name}.h5ad")

        # print(adata_concat.obsm.keys())
    ## done with adata_concat for one magnitude
    return adata_concat


def plot_titration_curves(results_df, out_folder):
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



if __name__ == "__main__":
    #magnitudes = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0]
    version = "test"
    magnitudes = [0.05, 0.1, 3.0]
    deltas = np.array([1.0, 2.0])
    out_folder = "delta_titration/results"
    adata_folder = "delta_titration/adata"
    results = []
    for output_name, mag in enumerate(magnitudes):
        # print(output_name)
        # print(mag)
        temp = mag * deltas
        # print(temp)
    #     
        _LOGGER.info(f"Magnitude = {mag}")
        _LOGGER.info(f"Deltas = {temp}")
        adata_concat = delta_magnitudes(out_folder = out_folder,
                                        adata_folder = adata_folder,
                                        output_name = output_name,
                                        deltas = deltas,
                                        mag = mag
                                        )
        # run baseline methods (original decipher, umap)
        latent_spaces = run_methods(adata_concat, seed=params.SEED)
        # Compute metrics on original Decipher
        # _LOGGER.info(f"Computing metrics on original Decipher...")
        # original_metrics = compute_batch_mixing_metrics(
        #     adata_concat,
        #     batch_key='delta',
        #     z_key='decipher_decipher_z'
        # )
        # original_metrics['magnitude'] = mag
        # original_metrics['max_delta'] = max(deltas)
        # original_metrics['increment'] = mag
        # original_metrics['method'] = 'original'
        
        # results.append(original_metrics)
        
        # train batch-corrected decipher
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

        config.initialize_from_adata(adata_concat, batch_key='delta')

        model, losses = train_batch_corrected_decipher(
            adata_concat,
            batch_key='delta',
            config=config,
            device='cpu'
        )

        # Extract batch-corrected embeddings
        bc_results = evaluate_batch_correction(model, adata_concat, batch_key='delta', device='cpu')
        adata_concat.obsm['X_decipher_batch_corrected_z'] = bc_results['z']
        adata_concat.obsm['X_decipher_batch_corrected_v'] = bc_results['v']
        adata_concat.obs['batch_attention_strength'] = bc_results['attention_weights'].mean(axis=(1, 2, 3))

        print(f"✓ Training complete (final loss: {losses['val_losses'][-1]:.2f})")

        # Compute metrics on batch-corrected embeddings
        print(f"\n[Step 5/5] Computing batch-corrected metrics...")
        bc_metrics = compute_batch_mixing_metrics(
            adata_concat,
            batch_key='delta',
            z_key='X_decipher_batch_corrected_z'
        )
        bc_metrics['magnitude'] = mag
        bc_metrics['max_delta'] = max(deltas)
        bc_metrics['increment'] = mag
        bc_metrics['method'] = 'batch_corrected'
        results.append(bc_metrics)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{out_folder}/titration_metrics_{version}.csv", index=False)
    print(f"Saved metrics: {out_folder}/titration_metrics_{version}.csv")
        
    plot_titration_curves(results_df, out_folder)
        
        
    
    
    