"""
Data loader utilities for batch-corrected Decipher model.

This module provides data loading functions that properly handle batch indices
for the attention-based batch correction mechanism.
"""

import numpy as np
import torch
import torch.utils.data


def get_dense_X(adata):
    """Convert AnnData.X to dense numpy array if sparse."""
    if isinstance(adata.X, np.ndarray):
        return adata.X
    else:
        return adata.X.toarray()


def make_batch_corrected_data_loader(
    adata,
    batch_key,
    batch_size=64,
    shuffle=True,
    **kwargs
):
    """
    Create a PyTorch DataLoader for batch-corrected Decipher model.

    This data loader returns:
    - Gene expression data (X)
    - Batch indices (as integers, not one-hot encoded)

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing gene expression and batch information
    batch_key : str
        Key in adata.obs containing batch information
    batch_size : int, default=64
        Batch size for training
    shuffle : bool, default=True
        Whether to shuffle data
    **kwargs
        Additional arguments passed to DataLoader

    Returns
    -------
    data_loader : torch.utils.data.DataLoader
        DataLoader that yields (genes, batch_indices) tuples
    batch_mapping : dict
        Mapping from original batch labels to integer indices
    """
    if batch_key not in adata.obs:
        raise ValueError(f"batch_key '{batch_key}' not found in adata.obs")

    # Get gene expression data
    genes = torch.FloatTensor(get_dense_X(adata))

    # Get batch indices as integers
    # Convert categorical to integer codes
    batch_categorical = adata.obs[batch_key].astype("category")
    batch_codes = batch_categorical.cat.codes.values
    batch_indices = torch.LongTensor(batch_codes)

    # Create mapping for reference
    batch_mapping = dict(enumerate(batch_categorical.cat.categories))

    # Create dataset and data loader
    dataset = torch.utils.data.TensorDataset(genes, batch_indices)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        **kwargs
    )

    return data_loader, batch_mapping


def make_train_val_loaders(
    adata,
    batch_key,
    batch_size=64,
    val_frac=0.1,
    seed=0,
    **kwargs
):
    """
    Create training and validation DataLoaders with proper splitting.

    Parameters
    ----------
    adata : AnnData
        Annotated data object
    batch_key : str
        Key in adata.obs containing batch information
    batch_size : int, default=64
        Batch size for training
    val_frac : float, default=0.1
        Fraction of data to use for validation
    seed : int, default=0
        Random seed for reproducible splitting
    **kwargs
        Additional arguments passed to DataLoader

    Returns
    -------
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    batch_mapping : dict
        Mapping from batch labels to integer indices
    """
    if batch_key not in adata.obs:
        raise ValueError(f"batch_key '{batch_key}' not found in adata.obs")

    # Get data
    genes = torch.FloatTensor(get_dense_X(adata))

    # Get batch indices
    batch_categorical = adata.obs[batch_key].astype("category")
    batch_codes = batch_categorical.cat.codes.values
    batch_indices = torch.LongTensor(batch_codes)

    # Create mapping
    batch_mapping = dict(enumerate(batch_categorical.cat.categories))

    # Create dataset
    dataset = torch.utils.data.TensorDataset(genes, batch_indices)

    # Split into train and validation
    n_total = len(dataset)
    n_val = int(n_total * val_frac)
    n_train = n_total - n_val

    # Set seed for reproducible split
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=generator
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **kwargs
    )

    return train_loader, val_loader, batch_mapping


# Example usage
if __name__ == "__main__":
    print("Testing batch-corrected data loaders...")

    # Create mock AnnData-like object
    class MockAnnData:
        def __init__(self, n_cells, n_genes):
            self.shape = (n_cells, n_genes)
            self.X = np.random.randn(n_cells, n_genes)
            self.obs = {
                'batch': ['batch_A', 'batch_B', 'batch_C'] * (n_cells // 3) + ['batch_A'] * (n_cells % 3)
            }

    # Create mock data
    n_cells = 100
    n_genes = 500
    mock_adata = MockAnnData(n_cells, n_genes)

    # Test single data loader
    print("\n1. Testing single data loader:")
    data_loader, batch_mapping = make_batch_corrected_data_loader(
        mock_adata,
        batch_key='batch',
        batch_size=16
    )

    print(f"   Batch mapping: {batch_mapping}")
    print(f"   Number of batches: {len(data_loader)}")

    # Get one batch
    genes_batch, batch_indices = next(iter(data_loader))
    print(f"   Sample batch shapes:")
    print(f"     genes: {genes_batch.shape}")
    print(f"     batch_indices: {batch_indices.shape}")
    print(f"     batch_indices values: {batch_indices[:5].tolist()}")

    # Test train/val split
    print("\n2. Testing train/val split:")
    train_loader, val_loader, batch_mapping = make_train_val_loaders(
        mock_adata,
        batch_key='batch',
        batch_size=16,
        val_frac=0.2,
        seed=42
    )

    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")

    # Count total samples
    train_samples = len(train_loader.dataset)
    val_samples = len(val_loader.dataset)
    print(f"   Training samples: {train_samples}")
    print(f"   Validation samples: {val_samples}")
    print(f"   Total: {train_samples + val_samples}")

    print("\n✓ All tests passed!")
