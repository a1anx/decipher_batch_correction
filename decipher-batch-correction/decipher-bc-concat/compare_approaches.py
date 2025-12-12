"""
Comparison script for Concatenation vs. Attention batch correction approaches.

This script trains both models on the same dataset and compares their performance.
"""

import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
import scanpy as sc

# Import concatenation approach
from decipher_batch_corrected_concat import (
    DecipherBatchCorrectedConcat,
    DecipherBatchCorrectedConcatConfig
)
from train_batch_corrected_concat import (
    train_batch_corrected_decipher_concat,
    evaluate_batch_correction_concat
)

# Import attention approach (from parent directory)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'decipher-bc'))

from decipher_batch_corrected import (
    DecipherBatchCorrected,
    DecipherBatchCorrectedConfig
)
from train_batch_corrected import (
    train_batch_corrected_decipher,
    evaluate_batch_correction
)


def count_parameters(model):
    """Count total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_batch_mixing_score(embeddings, batch_labels):
    """
    Compute batch mixing score using silhouette coefficient.
    Lower is better (indicates better batch mixing).
    """
    try:
        score = silhouette_score(embeddings, batch_labels)
        # Return 1 - score so that lower is better
        return 1 - score
    except:
        return np.nan


def compare_models(
    adata,
    batch_key='batch',
    device='cuda' if torch.cuda.is_available() else 'cpu',
    n_epochs=50
):
    """
    Train and compare both batch correction approaches.

    Parameters
    ----------
    adata : AnnData
        Input data with batch information
    batch_key : str
        Column in adata.obs containing batch labels
    device : str
        Device to train on
    n_epochs : int
        Number of training epochs

    Returns
    -------
    results : dict
        Comparison results including timing, performance metrics, and models
    """
    print("="*80)
    print("COMPARING BATCH CORRECTION APPROACHES")
    print("="*80)
    print(f"Dataset: {adata.shape[0]} cells, {adata.shape[1]} genes")
    print(f"Batches: {adata.obs[batch_key].nunique()}")
    print(f"Device: {device}")
    print("="*80)

    results = {
        'concat': {},
        'attention': {}
    }

    # Shared configuration parameters
    shared_params = {
        'dim_z': 10,
        'dim_v': 2,
        'batch_emb_dim': 64,
        'decoder_hidden_dims': [128, 256],
        'learning_rate': 5e-3,
        'batch_size': 64,
        'n_epochs': n_epochs,
        'early_stopping_patience': 10
    }

    # ========== TRAIN CONCATENATION MODEL ==========
    print("\n" + "="*80)
    print("1. CONCATENATION APPROACH")
    print("="*80)

    config_concat = DecipherBatchCorrectedConcatConfig(**shared_params)

    start_time = time.time()
    model_concat, losses_concat = train_batch_corrected_decipher_concat(
        adata,
        batch_key=batch_key,
        config=config_concat,
        device=device
    )
    concat_train_time = time.time() - start_time

    # Evaluate
    eval_results_concat = evaluate_batch_correction_concat(
        model_concat, adata, batch_key=batch_key, device=device
    )

    # Compute metrics
    batch_labels = adata.obs[batch_key].astype('category').cat.codes.values
    batch_mixing_concat = compute_batch_mixing_score(
        eval_results_concat['z'], batch_labels
    )

    results['concat'] = {
        'model': model_concat,
        'losses': losses_concat,
        'embeddings': eval_results_concat,
        'train_time': concat_train_time,
        'n_params': count_parameters(model_concat),
        'batch_mixing': batch_mixing_concat,
        'final_train_loss': losses_concat['train_losses'][-1],
        'final_val_loss': losses_concat['val_losses'][-1],
    }

    print(f"\nConcatenation Results:")
    print(f"  Training time: {concat_train_time:.1f}s")
    print(f"  Parameters: {results['concat']['n_params']:,}")
    print(f"  Final train loss: {results['concat']['final_train_loss']:.2f}")
    print(f"  Final val loss: {results['concat']['final_val_loss']:.2f}")
    print(f"  Batch mixing score: {batch_mixing_concat:.3f}")

    # ========== TRAIN ATTENTION MODEL ==========
    print("\n" + "="*80)
    print("2. ATTENTION APPROACH")
    print("="*80)

    config_attn = DecipherBatchCorrectedConfig(**shared_params)
    config_attn.n_attention_heads = 4
    config_attn.combination_mode = "concat"

    start_time = time.time()
    model_attn, losses_attn = train_batch_corrected_decipher(
        adata,
        batch_key=batch_key,
        config=config_attn,
        device=device
    )
    attn_train_time = time.time() - start_time

    # Evaluate
    eval_results_attn = evaluate_batch_correction(
        model_attn, adata, batch_key=batch_key, device=device
    )

    # Compute metrics
    batch_mixing_attn = compute_batch_mixing_score(
        eval_results_attn['z'], batch_labels
    )

    results['attention'] = {
        'model': model_attn,
        'losses': losses_attn,
        'embeddings': eval_results_attn,
        'train_time': attn_train_time,
        'n_params': count_parameters(model_attn),
        'batch_mixing': batch_mixing_attn,
        'final_train_loss': losses_attn['train_losses'][-1],
        'final_val_loss': losses_attn['val_losses'][-1],
    }

    print(f"\nAttention Results:")
    print(f"  Training time: {attn_train_time:.1f}s")
    print(f"  Parameters: {results['attention']['n_params']:,}")
    print(f"  Final train loss: {results['attention']['final_train_loss']:.2f}")
    print(f"  Final val loss: {results['attention']['final_val_loss']:.2f}")
    print(f"  Batch mixing score: {batch_mixing_attn:.3f}")

    # ========== COMPARISON SUMMARY ==========
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)

    print(f"\nTraining Speed:")
    print(f"  Concatenation: {concat_train_time:.1f}s")
    print(f"  Attention:     {attn_train_time:.1f}s")
    print(f"  Speedup:       {attn_train_time/concat_train_time:.2f}x")

    print(f"\nModel Size:")
    print(f"  Concatenation: {results['concat']['n_params']:,} params")
    print(f"  Attention:     {results['attention']['n_params']:,} params")
    print(f"  Difference:    {results['attention']['n_params'] - results['concat']['n_params']:,} params")

    print(f"\nBatch Mixing (lower is better):")
    print(f"  Concatenation: {batch_mixing_concat:.3f}")
    print(f"  Attention:     {batch_mixing_attn:.3f}")
    if batch_mixing_attn < batch_mixing_concat:
        print(f"  Winner: Attention (better by {batch_mixing_concat - batch_mixing_attn:.3f})")
    else:
        print(f"  Winner: Concatenation (better by {batch_mixing_attn - batch_mixing_concat:.3f})")

    print(f"\nValidation Loss:")
    print(f"  Concatenation: {results['concat']['final_val_loss']:.2f}")
    print(f"  Attention:     {results['attention']['final_val_loss']:.2f}")

    print("\n" + "="*80)

    return results


def plot_comparison(results, save_path=None):
    """
    Create comparison plots.

    Parameters
    ----------
    results : dict
        Results from compare_models()
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Training curves
    ax = axes[0, 0]
    epochs_concat = range(1, len(results['concat']['losses']['train_losses']) + 1)
    epochs_attn = range(1, len(results['attention']['losses']['train_losses']) + 1)

    ax.plot(epochs_concat, results['concat']['losses']['train_losses'],
            label='Concat Train', color='blue', alpha=0.7)
    ax.plot(epochs_concat, results['concat']['losses']['val_losses'],
            label='Concat Val', color='blue', linestyle='--', alpha=0.7)
    ax.plot(epochs_attn, results['attention']['losses']['train_losses'],
            label='Attn Train', color='red', alpha=0.7)
    ax.plot(epochs_attn, results['attention']['losses']['val_losses'],
            label='Attn Val', color='red', linestyle='--', alpha=0.7)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Parameter count comparison
    ax = axes[0, 1]
    methods = ['Concatenation', 'Attention']
    params = [results['concat']['n_params'], results['attention']['n_params']]
    bars = ax.bar(methods, params, color=['blue', 'red'], alpha=0.7)
    ax.set_ylabel('Number of Parameters')
    ax.set_title('Model Size Comparison')
    ax.ticklabel_format(style='plain', axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom')

    # Plot 3: Batch mixing comparison
    ax = axes[1, 0]
    methods = ['Concatenation', 'Attention']
    mixing = [results['concat']['batch_mixing'], results['attention']['batch_mixing']]
    bars = ax.bar(methods, mixing, color=['blue', 'red'], alpha=0.7)
    ax.set_ylabel('Batch Mixing Score (lower is better)')
    ax.set_title('Batch Mixing Performance')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')

    # Plot 4: Training time comparison
    ax = axes[1, 1]
    methods = ['Concatenation', 'Attention']
    times = [results['concat']['train_time'], results['attention']['train_time']]
    bars = ax.bar(methods, times, color=['blue', 'red'], alpha=0.7)
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Training Speed Comparison')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s',
                ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved comparison plot to: {save_path}")

    return fig


# Example usage
if __name__ == "__main__":
    print("Creating synthetic data for comparison...")

    n_cells = 2000
    n_genes = 500
    n_batches = 3

    # Create synthetic AnnData with batch effects
    X = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes))

    # Add batch effects (different means for different batches)
    batch_labels = np.array([f'batch_{i % n_batches}' for i in range(n_cells)])
    for i, batch in enumerate([f'batch_{j}' for j in range(n_batches)]):
        mask = batch_labels == batch
        X[mask] = X[mask] * (1.0 + 0.3 * i)  # Add batch effect

    adata = sc.AnnData(X=X)
    adata.obs['batch'] = batch_labels
    adata.var_names = [f'gene_{i}' for i in range(n_genes)]

    print(f"Created synthetic data: {adata.shape}")
    print(f"Batches: {adata.obs['batch'].value_counts().to_dict()}")

    # Run comparison
    results = compare_models(
        adata,
        batch_key='batch',
        device='cpu',
        n_epochs=30  # Reduce for quick testing
    )

    # Create comparison plots
    fig = plot_comparison(results, save_path='comparison_concat_vs_attention.png')
    plt.show()

    # Add embeddings to adata for further analysis
    adata.obsm['X_decipher_z_concat'] = results['concat']['embeddings']['z']
    adata.obsm['X_decipher_v_concat'] = results['concat']['embeddings']['v']
    adata.obsm['X_decipher_z_attn'] = results['attention']['embeddings']['z']
    adata.obsm['X_decipher_v_attn'] = results['attention']['embeddings']['v']

    print("\n✓ Comparison complete!")
    print("\nRecommendation:")
    if results['attention']['batch_mixing'] < results['concat']['batch_mixing']:
        improvement = results['concat']['batch_mixing'] - results['attention']['batch_mixing']
        if improvement > 0.05:
            print("  → Use ATTENTION approach (significantly better batch mixing)")
        else:
            print("  → Both approaches perform similarly")
            print("  → Consider CONCATENATION for faster training")
    else:
        print("  → Use CONCATENATION approach (simpler and faster)")
