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
from simulations import simulation_correlated_shift, run_methods
import params
from params import SIMUL_PARAMS



parent_dir = Path(__file__).parent.parent / "decipher-batch-correction" / "decipher-bc"
sys.path.insert(0, str(parent_dir))
from decipher_batch_corrected import DecipherBatchCorrectedConfig
from train_batch_corrected import train_batch_corrected_decipher, evaluate_batch_correction


_LOGGER = logging.getLogger(__name__)


def compute_batch_mixing_metrics(adata, batch_key='shift', z_key='X_decipher_batch_corrected_z'):
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
    #if 'batch_attention_strength' in adata.obs.columns:
    #    metrics['mean_attention_strength'] = adata.obs['batch_attention_strength'].mean()

    return metrics


def shift_magnitudes(
    adata_folder,
    output_name,
    shift_type,
    shifts,
    mag
):
    """
    Compute comprehensive metrics to assess batch mixing quality.
    
    Calls:
    - simulation_correlated_shift from simulations
    - run_methods from simulations

    Metrics:
    - batch_silhouette: Lower = better batch mixing
    - pseudotime_correlation: Higher = better biological preservation
    - cluster_ari: Higher = better biological structure preservation
    - mean_attention_strength: Model's batch correction effort

    Returns:
        adata_concat: concatenated adata of unshifted base data and shifted adatas
    """
    os.makedirs(adata_folder, exist_ok=True)
    shift_vec = mag * shifts
    
    # Unshifted Base (shift=0)
    adata_base = simulation_correlated_shift(
        **SIMUL_PARAMS,
        shift_type="none",
        shift=0,
    )
    ground_truth_clusters = adata_base.obs["cluster_true"]
    ground_truth_cluster_rank = adata_base.uns["cluster_rank"] 
    
    # Run all methods on the dataset
    latent_spaces = run_methods(adata_base, seed=params.SEED)
    # _LOGGER.info(f"Latent spaces: {latent_spaces}")
    # adata_base.write(f"{adata_folder}/adata_base.h5ad")
    # _LOGGER.info(f"Adata saved: {adata_folder}/adata_base.h5ad")
    
    
    adata_concat = adata_base.copy()

    # Shifted adatas
    for shift in shift_vec:
        adata_sim = simulation_correlated_shift(
            **SIMUL_PARAMS,
            shift_type=shift_type,
            shift=shift,
        )
        # override the clusters
        adata_sim.obs["cluster_true"] = ground_truth_clusters
        adata_sim.uns["cluster_rank"] = ground_truth_cluster_rank
        
        # 4Run all methods on the dataset
        latent_spaces = run_methods(adata_sim, seed=params.SEED)
        
        adata_concat = ad.concat([adata_concat, adata_sim], axis=0,
                                    join="outer", label=None, merge="same")

    adata_concat.write(os.path.join(adata_folder, f"adata_combined_{shift_type}_{output_name}.h5ad"))
    _LOGGER.info(f"Combined adata saved: {adata_folder}/adata_combined_{shift_type}_{output_name}.h5ad")

    return adata_concat


def plot_titration_curves(results_df, out_folder, shift_type):
    """
    Plot titration curves showing batch correction performance vs magnitude.
    
    """
    os.makedirs(out_folder, exist_ok=True)
    sns.set_style("whitegrid")

    # Create comprehensive figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics_to_plot = [
        ('batch_silhouette', 'Batch Silhouette Score', 'Lower = Better Mixing'),
        ('pseudotime_correlation', 'Pseudotime Correlation', 'Higher = Better Biological Preservation'),
        ('cluster_ari', 'Cluster ARI', 'Higher = Better Structure Preservation'),
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
        #ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(f"{out_folder}/titration_curves_{shift_type}_all_metrics.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {out_folder}/titration_curves_{shift_type}_all_metrics.png")
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
        #ax.set_xscale('log')

        # Shade region where improvement is positive
        ax.fill_between(improvement.index, 0, improvement.values,
                        where=(improvement.values > 0), alpha=0.2, color='green',
                        label='Effective Batch Correction')

        plt.tight_layout()
        plt.savefig(f"{out_folder}/{shift_type}_improvement_titration.png", dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {out_folder}/{shift_type}_improvement_titration.png")
        plt.close()


def run_titration(
    shift_type,
    magnitudes,
    shifts,
    out_folder,
    adata_folder,
):
    """
    Creates one or more sets of shifted, concatenated datasets with differing magnitudes
    
    Calls:
    - shift_magnitudes
    - run_methods
    - compute_batch_mixing_metrics
    - DecipherBatchCorrectedConfig
    - train_batch_corrected_decipher
    - evaluate_batch_correction
    
    
    """
    results = []
    for mag in magnitudes:
        # print(output_name)
        # print(mag)
        temp = [0]
        temp = temp + list(mag * shifts)
        # print(temp)
        output_name = str(temp)
        _LOGGER.info(f"Magnitude = {mag}")
        _LOGGER.info(f"{shift_type} = {temp}")
        adata_concat = shift_magnitudes(adata_folder = adata_folder,
                                        output_name = output_name,
                                        shift_type = shift_type,
                                        shifts = shifts,
                                        mag = mag
                                        )
        # run baseline methods (original decipher, umap)
        latent_spaces = run_methods(adata_concat, seed=params.SEED)
        
        # Compute metrics on original Decipher
        _LOGGER.info(f"Computing metrics on original Decipher...")
        original_metrics = compute_batch_mixing_metrics(
            adata_concat,
            batch_key='shift',
            z_key='decipher_decipher_z'
        )
        original_metrics['magnitude'] = mag
        original_metrics['max_shift'] = max(shifts)
        original_metrics['increment'] = mag
        original_metrics['method'] = 'original'
        
        results.append(original_metrics)
        
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

        config.initialize_from_adata(adata_concat, batch_key='shift')

        model, losses = train_batch_corrected_decipher(
            adata_concat,
            batch_key='shift',
            config=config,
            device='cpu'
        )

        # Extract batch-corrected embeddings
        bc_results = evaluate_batch_correction(model, adata_concat, batch_key='shift', device='cpu')
        adata_concat.obsm['X_decipher_batch_corrected_z'] = bc_results['z']
        adata_concat.obsm['X_decipher_batch_corrected_v'] = bc_results['v']
        adata_concat.obs['batch_attention_strength'] = bc_results['attention_weights'].mean(axis=(1, 2, 3))

        _LOGGER.info(f"Training complete (final loss: {losses['val_losses'][-1]:.2f})")
    
        # Compute metrics on batch-corrected embeddings
        bc_metrics = compute_batch_mixing_metrics(
            adata_concat,
            batch_key='shift',
            z_key='X_decipher_batch_corrected_z'
        )
        bc_metrics['magnitude'] = mag
        bc_metrics['max_shift'] = max(shifts)
        bc_metrics['increment'] = mag
        bc_metrics['method'] = 'batch_corrected'
        results.append(bc_metrics)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{out_folder}/{shift_type}_titration_metrics_{version}.csv", index=False)
    print(f"Saved metrics: {out_folder}/{shift_type}_titration_metrics_{version}.csv")
    return results_df   
        
    


if __name__ == "__main__":
    # (1) Variables for Native Decipher vs Decipher-BC comparison dataset
    version = "simul"
    shifts = np.array([0.01, 0.05, 0.1])
    adata_folder = "simulation/concat_adata"
    output_name = str(shifts)
    shift_magnitudes(adata_folder,
                     output_name ,
                     shift_type = "alpha",
                     shifts = shifts,
                     mag = 1)
    shift_magnitudes(adata_folder,
                     output_name ,
                     shift_type = "delta",
                     shifts = shifts,
                     mag = 1)
    # (2) Titration
    # magnitudes = [0.05, 0.02, 0.1, 0.2, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
    # version = "1"
    # shifts = np.array([1.0, 2.0, 3.0])
    # out_folder = "titration/results"
    # adata_folder = "titration/adata"
    
    # alpha
    # shift_type = "alpha"
    # results_df = run_titration(shift_type,
    #               magnitudes,
    #               shifts,
    #               out_folder,
    #               adata_folder,
    #               )
        
    # # plot_titration_curves(results_df, out_folder, shift_type="alpha")
    
    # # delta
    # shift_type = "delta"
    # results_df = run_titration(shift_type,
    #               magnitudes,
    #               shifts,
    #               out_folder,
    #               adata_folder,)
        
    # plot_titration_curves(results_df, out_folder, shift_type="delta")
    
        
        
    
    
    