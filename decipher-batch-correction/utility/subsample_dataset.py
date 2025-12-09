"""
Subsample large scRNA-seq dataset for manageable analysis.

This script creates a stratified subsample of the BoneMarrowMap dataset
that preserves:
1. Donor distribution (for batch effects)
2. Cell type diversity (for biological trajectories)
3. Manageable size for VM processing
"""

import scanpy as sc
import numpy as np
import argparse


def subsample_adata(
    input_path,
    output_path,
    n_cells_per_donor=2000,
    stratify_by='cell_type',
    seed=42
):
    """
    Create a stratified subsample of the dataset.

    Parameters
    ----------
    input_path : str
        Path to the full h5ad file
    n_cells_per_donor : int
        Target number of cells per donor (adjust based on VM memory)
    stratify_by : str
        Column to stratify by (e.g., 'cell_type', 'celltype', 'leiden')
    seed : int
        Random seed for reproducibility
    """
    print(f"Loading dataset from {input_path}...")
    print("(This may take several minutes for large files)")

    # Load the full dataset
    # Use backed='r' mode to avoid loading entire dataset into memory
    adata = sc.read_h5ad(input_path, backed='r')

    print(f"Original dataset: {adata.shape[0]} cells x {adata.shape[1]} genes")
    print(f"Memory mode: {type(adata.X)}")

    # Check what metadata is available
    print(f"\nAvailable metadata columns: {list(adata.obs.columns)}")

    # Identify batch/donor column
    donor_cols = [col for col in adata.obs.columns if 'donor' in col.lower() or 'batch' in col.lower() or 'sample' in col.lower()]
    if donor_cols:
        print(f"\nPotential donor/batch columns: {donor_cols}")
        batch_key = donor_cols[0]
    else:
        print("\nWarning: No obvious donor/batch column found!")
        print("You may need to manually specify the batch_key parameter")
        batch_key = input("Enter the batch/donor column name: ")

    # Load into memory in chunks and subsample
    print(f"\nSubsampling {n_cells_per_donor} cells per {batch_key}...")

    # Get donor/batch information without loading full matrix
    donors = adata.obs[batch_key].unique()
    print(f"Found {len(donors)} unique donors/batches")

    # Stratified sampling per donor
    np.random.seed(seed)
    indices_to_keep = []

    for donor in donors:
        donor_mask = adata.obs[batch_key] == donor
        donor_indices = np.where(donor_mask)[0]

        # If stratify_by column exists, do stratified sampling
        if stratify_by in adata.obs.columns:
            # Sample proportionally from each cell type within this donor
            cell_types = adata.obs[stratify_by][donor_mask].value_counts()

            for cell_type, count in cell_types.items():
                # Proportional sampling
                n_sample = int(n_cells_per_donor * count / len(donor_indices))
                n_sample = max(1, n_sample)  # At least 1 cell per type

                ct_indices = donor_indices[adata.obs[stratify_by][donor_mask] == cell_type]

                if len(ct_indices) > n_sample:
                    sampled = np.random.choice(ct_indices, size=n_sample, replace=False)
                else:
                    sampled = ct_indices

                indices_to_keep.extend(sampled)
        else:
            # Simple random sampling per donor
            if len(donor_indices) > n_cells_per_donor:
                sampled = np.random.choice(donor_indices, size=n_cells_per_donor, replace=False)
            else:
                sampled = donor_indices

            indices_to_keep.extend(sampled)

    print(f"\nTotal cells after subsampling: {len(indices_to_keep)}")

    # Load only the subsampled cells
    adata_subset = adata[indices_to_keep].to_memory()

    # Print summary statistics
    print(f"\nSubsampled dataset: {adata_subset.shape[0]} cells x {adata_subset.shape[1]} genes")
    print(f"Donors represented: {adata_subset.obs[batch_key].nunique()}")

    if stratify_by in adata_subset.obs.columns:
        print(f"\nCell type distribution:")
        print(adata_subset.obs[stratify_by].value_counts())

    # Estimate memory size
    size_mb = (adata_subset.X.data.nbytes if hasattr(adata_subset.X, 'data') else adata_subset.X.nbytes) / (1024**2)
    print(f"\nEstimated memory usage: {size_mb:.2f} MB")

    # Save subsampled dataset
    print(f"\nSaving subsampled dataset to {output_path}...")
    adata_subset.write_h5ad(output_path)

    print("✓ Done!")
    return adata_subset


def quick_explore_dataset(input_path):
    """
    Quickly explore dataset structure without loading full data.
    """
    print(f"Exploring dataset structure: {input_path}")

    adata = sc.read_h5ad(input_path, backed='r')

    print(f"\nDataset dimensions: {adata.shape[0]} cells x {adata.shape[1]} genes")
    print(f"\nMetadata columns (adata.obs):")
    for col in adata.obs.columns:
        n_unique = adata.obs[col].nunique()
        print(f"  - {col}: {n_unique} unique values")
        if n_unique < 20:
            print(f"    Values: {list(adata.obs[col].unique())}")

    print(f"\nVariable annotations (adata.var):")
    print(f"  Columns: {list(adata.var.columns)}")

    print(f"\nUnstructured annotations (adata.uns):")
    print(f"  Keys: {list(adata.uns.keys())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample large scRNA-seq dataset")
    parser.add_argument("input", help="Path to input h5ad file")
    parser.add_argument("-o", "--output", help="Path to output h5ad file (default: input_subsampled.h5ad)")
    parser.add_argument("-n", "--n-cells", type=int, default=2000, help="Target cells per donor (default: 2000)")
    parser.add_argument("--stratify", default="cell_type", help="Column to stratify by (default: cell_type)")
    parser.add_argument("--explore", action="store_true", help="Just explore dataset structure without subsampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    if args.explore:
        quick_explore_dataset(args.input)
    else:
        if args.output is None:
            args.output = args.input.replace('.h5ad', '_subsampled.h5ad')

        subsample_adata(
            args.input,
            args.output,
            n_cells_per_donor=args.n_cells,
            stratify_by=args.stratify,
            seed=args.seed
        )
