"""
Training script for concatenation-based batch-corrected Decipher model.

This script demonstrates how to train the DecipherBatchCorrectedConcat model
using Pyro's Stochastic Variational Inference (SVI).

This is a simpler baseline approach compared to the attention-based model.
"""

import logging
import numpy as np
import torch
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

from decipher_batch_corrected_concat import DecipherBatchCorrectedConcat, DecipherBatchCorrectedConcatConfig
from data_loader_batch_corrected import make_train_val_loaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_batch_corrected_decipher_concat(
    adata,
    batch_key,
    config=None,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Train a concatenation-based batch-corrected Decipher model.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with gene expression and batch information
    batch_key : str
        Key in adata.obs containing batch information
    config : DecipherBatchCorrectedConcatConfig, optional
        Model configuration. If None, uses default configuration.
    device : str, default='cuda' if available else 'cpu'
        Device to train on

    Returns
    -------
    model : DecipherBatchCorrectedConcat
        Trained model
    losses : dict
        Dictionary containing training and validation losses
    """
    # Initialize configuration if not provided
    if config is None:
        config = DecipherBatchCorrectedConcatConfig()

    # Initialize from data
    config.initialize_from_adata(adata, batch_key=batch_key)

    logger.info(f"Initialized config: n_cells={config.n_cells}, "
                f"n_genes={config.dim_genes}, n_batches={config.n_batches}")
    logger.info(f"Using CONCATENATION-based batch correction (simpler baseline)")

    # Create data loaders
    train_loader, val_loader, batch_mapping = make_train_val_loaders(
        adata,
        batch_key=batch_key,
        batch_size=config.batch_size,
        val_frac=config.val_frac,
        seed=config.seed
    )

    logger.info(f"Batch mapping: {batch_mapping}")
    logger.info(f"Training samples: {len(train_loader.dataset)}, "
                f"Validation samples: {len(val_loader.dataset)}")

    # Initialize model
    model = DecipherBatchCorrectedConcat(config).to(device)
    logger.info(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    # Setup Pyro optimizer and inference
    optimizer = ClippedAdam({
        "lr": config.learning_rate,
        "clip_norm": 10.0
    })

    svi = SVI(
        model.model,
        model.guide,
        optimizer,
        loss=Trace_ELBO()
    )

    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    logger.info("Starting training...")

    for epoch in range(config.n_epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        n_train_batches = 0

        for genes_batch, batch_indices in train_loader:
            genes_batch = genes_batch.to(device)
            batch_indices = batch_indices.to(device)

            # SVI step
            loss = svi.step(genes_batch, batch_indices)
            epoch_loss += loss
            n_train_batches += 1

        avg_train_loss = epoch_loss / n_train_batches
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for genes_batch, batch_indices in val_loader:
                genes_batch = genes_batch.to(device)
                batch_indices = batch_indices.to(device)

                # Evaluate loss
                loss = svi.evaluate_loss(genes_batch, batch_indices)
                val_loss += loss
                n_val_batches += 1

        avg_val_loss = val_loss / n_val_batches
        val_losses.append(avg_val_loss)

        # Logging
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch + 1}/{config.n_epochs} - "
                f"Train Loss: {avg_train_loss:.2f}, "
                f"Val Loss: {avg_val_loss:.2f}"
            )

        # Early stopping
        if config.early_stopping_patience is not None:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model state
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                # Restore best model
                model.load_state_dict(best_model_state)
                break

    logger.info("Training complete!")

    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'batch_mapping': batch_mapping
    }


def evaluate_batch_correction_concat(model, adata, batch_key, device='cpu'):
    """
    Evaluate the batch correction for concatenation approach.

    Parameters
    ----------
    model : DecipherBatchCorrectedConcat
        Trained model
    adata : AnnData
        Data to evaluate
    batch_key : str
        Key in adata.obs for batch information
    device : str
        Device to use

    Returns
    -------
    results : dict
        Dictionary containing evaluation metrics and embeddings
    """
    from data_loader_batch_corrected import make_batch_corrected_data_loader

    model.eval()
    model = model.to(device)

    # Create data loader
    data_loader, batch_mapping = make_batch_corrected_data_loader(
        adata,
        batch_key=batch_key,
        batch_size=128,
        shuffle=False
    )

    all_z_locs = []
    all_v_locs = []
    all_batch_indices = []

    with torch.no_grad():
        for genes_batch, batch_indices in data_loader:
            genes_batch = genes_batch.to(device)
            batch_indices = batch_indices.to(device)

            # Get latent representations
            z_loc, v_loc, _, _ = model.guide(genes_batch, batch_indices)

            all_z_locs.append(z_loc.cpu().numpy())
            all_v_locs.append(v_loc.cpu().numpy())
            all_batch_indices.append(batch_indices.cpu().numpy())

    # Concatenate results
    z = np.concatenate(all_z_locs, axis=0)
    v = np.concatenate(all_v_locs, axis=0)
    batch_indices = np.concatenate(all_batch_indices, axis=0)

    logger.info(f"Computed embeddings: z shape={z.shape}, v shape={v.shape}")

    return {
        'z': z,
        'v': v,
        'batch_indices': batch_indices,
        'batch_mapping': batch_mapping
    }


# Example usage
if __name__ == "__main__":
    import scanpy as sc

    # Load example data (replace with your own data)
    # For testing, we'll create synthetic data
    print("="*60)
    print("CONCATENATION-BASED BATCH CORRECTION")
    print("="*60)
    print("Creating synthetic data for testing...")

    n_cells = 1000
    n_genes = 500
    n_batches = 3

    # Create synthetic AnnData
    X = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes))
    adata = sc.AnnData(X=X)
    adata.obs['batch'] = [f'batch_{i % n_batches}' for i in range(n_cells)]
    adata.var_names = [f'gene_{i}' for i in range(n_genes)]

    print(f"Created synthetic AnnData: {adata.shape}")
    print(f"Batches: {adata.obs['batch'].value_counts()}")

    # Configure model
    config = DecipherBatchCorrectedConcatConfig(
        dim_z=10,
        dim_v=2,
        batch_emb_dim=32,
        decoder_hidden_dims=[64, 128],
        learning_rate=5e-3,
        batch_size=64,
        n_epochs=50,
        early_stopping_patience=10
    )

    # Train model
    print("\nTraining model...")
    model, losses = train_batch_corrected_decipher_concat(
        adata,
        batch_key='batch',
        config=config,
        device='cpu'
    )

    # Evaluate
    print("\nEvaluating batch correction...")
    results = evaluate_batch_correction_concat(model, adata, batch_key='batch')

    # Add embeddings to AnnData
    adata.obsm['X_decipher_z_concat'] = results['z']
    adata.obsm['X_decipher_v_concat'] = results['v']

    print(f"\nAdded embeddings to adata.obsm:")
    print(f"  X_decipher_z_concat: {adata.obsm['X_decipher_z_concat'].shape}")
    print(f"  X_decipher_v_concat: {adata.obsm['X_decipher_v_concat'].shape}")

    print("\n" + "="*60)
    print("SUMMARY: Concatenation-based approach")
    print("="*60)
    print("This approach uses: concat([z, batch_embedding]) -> MLP")
    print("Simpler than attention, good baseline for comparison")
    print("="*60)
    print("\n✓ Training and evaluation complete!")
