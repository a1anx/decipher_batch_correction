"""
Preprocessing script for BoneMarrowMap dataset - SUBSET VERSION for local machine

This script:
1. Loads the full BoneMarrowMap dataset
2. Explores metadata structure (donors, cell types, etc.)
3. Creates SUBSAMPLED versions (small: 20k cells, medium: 90k cells)
4. Filters to top 5k highly variable genes (HVGs)
5. Saves preprocessed datasets for training on local machine

The HVG selection preserves batch effects while reducing computational cost.
Subsampling maintains donor representation and cell type diversity.
"""

import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

print("=" * 80)
print("BONEMARROWMAP PREPROCESSING - Subset Version (Local Machine)")
print("=" * 80)

# Configuration
N_CELLS_SMALL = 20000    # Small subset: ~440 cells per donor (45 donors)
N_CELLS_MEDIUM = 90000   # Medium subset: ~2000 cells per donor (45 donors)
N_GENES = 5000          # Number of highly variable genes

# Step 1: Load dataset
print("\n[1/6] Loading BoneMarrowMap dataset...")
print("Note: This may take several minutes for large files")

input_file = "BoneMarrowMap_Annotated_Dataset_expandedFeatures.h5ad"

if not os.path.exists(input_file):
    print(f"\n❌ ERROR: File not found: {input_file}")
    print(f"   Current directory: {os.getcwd()}")
    print(f"   Please ensure the file is in: {os.path.abspath('.')}")
    sys.exit(1)

# Load with backed mode to avoid loading entire matrix into memory
print(f"   Loading from: {input_file}")
adata = sc.read_h5ad(input_file, backed='r')

print(f"✓ Loaded dataset: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
print(f"  Memory mode: backed (read-only, low memory)")

# Step 2: Explore metadata
print("\n[2/6] Exploring metadata structure...")
print(f"\nAvailable metadata columns in adata.obs:")
for i, col in enumerate(adata.obs.columns, 1):
    n_unique = adata.obs[col].nunique()
    print(f"  {i:2d}. {col:30s} - {n_unique:5d} unique values")

    # Show values if few unique
    if n_unique <= 10:
        unique_vals = list(adata.obs[col].unique()[:10])
        print(f"      Values: {unique_vals}")

# Identify batch/donor column
print("\n🔍 Identifying batch/donor column...")
donor_cols = [col for col in adata.obs.columns
              if 'donor' in col.lower() or 'batch' in col.lower()
              or 'sample' in col.lower() or 'patient' in col.lower()]

if donor_cols:
    print(f"✓ Found potential batch columns: {donor_cols}")
    batch_key = donor_cols[0]
    print(f"  Using: '{batch_key}'")
else:
    print("⚠ Warning: No obvious batch/donor column found!")
    print("  Common column names to look for:")
    print("    - 'donor', 'donor_id', 'DonorID'")
    print("    - 'batch', 'sample', 'patient'")
    print("\n  Please manually select from the list above:")
    batch_key = input("  Enter batch column name: ").strip()

n_batches = adata.obs[batch_key].nunique()
print(f"\n✓ Found {n_batches} unique batches/donors")

# Show batch distribution
print("\nBatch/Donor distribution (top 20):")
batch_counts = adata.obs[batch_key].value_counts().head(20)
for batch_id, count in batch_counts.items():
    print(f"  {str(batch_id):30s}: {count:6,} cells")

if len(batch_counts) > 20:
    print(f"  ... and {n_batches - 20} more batches")

# Identify cell type column
print("\n🔍 Identifying cell type column...")
cell_type_cols = [col for col in adata.obs.columns
                  if 'type' in col.lower() or 'cluster' in col.lower()
                  or 'celltype' in col.lower() or 'annotation' in col.lower()]

if cell_type_cols:
    print(f"✓ Found potential cell type columns: {cell_type_cols}")
    cell_type_key = cell_type_cols[0]
    print(f"  Using: '{cell_type_key}'")

    n_types = adata.obs[cell_type_key].nunique()
    print(f"\n✓ Found {n_types} unique cell types")

    print("\nCell type distribution (top 15):")
    type_counts = adata.obs[cell_type_key].value_counts().head(15)
    for cell_type, count in type_counts.items():
        print(f"  {str(cell_type):40s}: {count:6,} cells")
else:
    print("⚠ No cell type column found - will proceed without cell type info")
    cell_type_key = None

# Step 3: Create subsampled datasets
print("\n" + "=" * 80)
print("CREATING SUBSAMPLED DATASETS")
print("=" * 80)

# Calculate cells per donor
cells_per_donor_small = N_CELLS_SMALL // n_batches
cells_per_donor_medium = N_CELLS_MEDIUM // n_batches

print(f"\nTarget subsample sizes:")
print(f"  SMALL:  {N_CELLS_SMALL:,} cells (~{cells_per_donor_small} cells/donor × {n_batches} donors)")
print(f"  MEDIUM: {N_CELLS_MEDIUM:,} cells (~{cells_per_donor_medium} cells/donor × {n_batches} donors)")

def create_stratified_subsample(adata_backed, batch_key, cell_type_key, n_cells_per_donor, subset_name, seed=42):
    """
    Create stratified subsample maintaining:
    - All donors represented
    - Cell type proportions preserved within each donor
    """
    print(f"\n[Subsampling: {subset_name}]")
    print(f"  Target: ~{n_cells_per_donor} cells per donor")

    np.random.seed(seed)
    indices_to_keep = []

    donors = adata_backed.obs[batch_key].unique()

    for donor in donors:
        donor_mask = adata_backed.obs[batch_key] == donor
        donor_indices = np.where(donor_mask)[0]

        if cell_type_key and cell_type_key in adata_backed.obs.columns:
            # Stratified sampling by cell type within donor
            cell_types = adata_backed.obs[cell_type_key][donor_mask].value_counts()

            for cell_type, count in cell_types.items():
                # Proportional sampling
                n_sample = int(n_cells_per_donor * count / len(donor_indices))
                n_sample = max(1, n_sample)  # At least 1 cell per type

                ct_mask = (adata_backed.obs[batch_key] == donor) & (adata_backed.obs[cell_type_key] == cell_type)
                ct_indices = np.where(ct_mask)[0]

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

    print(f"  Sampled {len(indices_to_keep):,} cells")

    # Load only subsampled cells into memory
    adata_subset = adata_backed[indices_to_keep].to_memory()

    print(f"  ✓ Loaded into memory: {adata_subset.shape[0]:,} cells × {adata_subset.shape[1]:,} genes")

    return adata_subset

# Create both subsets
print("\n[3/6] Creating SMALL subset...")
adata_small = create_stratified_subsample(adata, batch_key, cell_type_key,
                                          cells_per_donor_small, "SMALL", seed=42)

print("\n[4/6] Creating MEDIUM subset...")
adata_medium = create_stratified_subsample(adata, batch_key, cell_type_key,
                                           cells_per_donor_medium, "MEDIUM", seed=42)

# Step 4: Filter to highly variable genes (apply to both subsets)
print("\n[5/6] Selecting highly variable genes...")
print(f"  Filtering to top {N_GENES:,} genes using Seurat v3 method")

def filter_to_hvg(adata_subset, batch_key, n_genes, subset_name):
    """Filter dataset to highly variable genes"""
    print(f"\n  [{subset_name}] Normalizing and computing HVGs...")

    # IMPORTANT: Save raw counts in a layer BEFORE normalization
    # The Decipher model uses NegativeBinomial distribution which requires integer counts
    adata_subset.layers['counts'] = adata_subset.X.copy()

    # Normalize
    sc.pp.normalize_total(adata_subset, target_sum=1e4)
    sc.pp.log1p(adata_subset)

    # Select HVGs with batch correction
    # Use 'seurat' flavor instead of 'seurat_v3' to avoid scikit-misc dependency
    sc.pp.highly_variable_genes(
        adata_subset,
        n_top_genes=n_genes,
        batch_key=batch_key,
        flavor='seurat',  # Changed from 'seurat_v3' to avoid scikit-misc
        subset=False
    )

    n_hvg = adata_subset.var['highly_variable'].sum()
    print(f"    Identified {n_hvg} highly variable genes")

    # Subset to HVGs
    adata_filtered = adata_subset[:, adata_subset.var['highly_variable']].copy()

    # Calculate size
    size_mb = (adata_filtered.X.data.nbytes if hasattr(adata_filtered.X, 'data')
               else adata_filtered.X.nbytes) / (1024**2)

    print(f"    ✓ Filtered: {adata_filtered.shape[0]:,} cells × {adata_filtered.shape[1]:,} genes ({size_mb:.1f} MB)")
    print(f"    ✓ Raw counts preserved in layer 'counts' for NegativeBinomial likelihood")

    return adata_filtered

adata_small_filtered = filter_to_hvg(adata_small, batch_key, N_GENES, "SMALL")
adata_medium_filtered = filter_to_hvg(adata_medium, batch_key, N_GENES, "MEDIUM")

# Step 5: Save filtered datasets
print("\n[6/6] Saving filtered datasets...")

output_small = "BoneMarrowMap_small_20k_5kgenes.h5ad"
output_medium = "BoneMarrowMap_medium_90k_5kgenes.h5ad"

adata_small_filtered.write_h5ad(output_small)
print(f"✓ SMALL saved to: {output_small}")

adata_medium_filtered.write_h5ad(output_medium)
print(f"✓ MEDIUM saved to: {output_medium}")

# Create summary file
print("\nCreating summary statistics file...")
summary_file = "BoneMarrowMap_subset_preprocessing_summary.txt"
with open(summary_file, 'w') as f:
    f.write("BoneMarrowMap Subset Preprocessing Summary\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"Original dataset: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes\n\n")

    f.write(f"SMALL subset:  {adata_small_filtered.shape[0]:,} cells × {adata_small_filtered.shape[1]:,} genes\n")
    f.write(f"MEDIUM subset: {adata_medium_filtered.shape[0]:,} cells × {adata_medium_filtered.shape[1]:,} genes\n\n")

    f.write(f"Batch column: {batch_key}\n")
    f.write(f"Number of batches: {n_batches}\n\n")

    if cell_type_key:
        f.write(f"Cell type column: {cell_type_key}\n")
        f.write(f"Number of cell types (small): {adata_small_filtered.obs[cell_type_key].nunique()}\n")
        f.write(f"Number of cell types (medium): {adata_medium_filtered.obs[cell_type_key].nunique()}\n\n")

    f.write("Subsampling strategy:\n")
    f.write(f"  - Stratified by donor and cell type\n")
    f.write(f"  - ~{cells_per_donor_small} cells/donor (small)\n")
    f.write(f"  - ~{cells_per_donor_medium} cells/donor (medium)\n\n")

    f.write(f"HVG selection method: Seurat v3 with batch correction\n")
    f.write(f"Number of HVGs: {N_GENES}\n")

print(f"✓ Summary saved to: {summary_file}")

# Print summary
print("\n" + "=" * 80)
print("PREPROCESSING COMPLETE")
print("=" * 80)

print(f"\n✓ Output files created:")

small_size_mb = os.path.getsize(output_small) / (1024**2)
medium_size_mb = os.path.getsize(output_medium) / (1024**2)

print(f"\n  1. {output_small}")
print(f"     - Small subset for quick testing")
print(f"     - {adata_small_filtered.shape[0]:,} cells × {adata_small_filtered.shape[1]:,} genes")
print(f"     - {small_size_mb:.1f} MB")

print(f"\n  2. {output_medium}")
print(f"     - Medium subset for full training")
print(f"     - {adata_medium_filtered.shape[0]:,} cells × {adata_medium_filtered.shape[1]:,} genes")
print(f"     - {medium_size_mb:.1f} MB")

print(f"\n  3. {summary_file}")
print(f"     - Summary statistics")

print(f"\n📊 Dataset comparison:")
print(f"  {'':30s} {'Original':>12s} {'Small':>12s} {'Medium':>12s}")
print(f"  {'-'*66}")
print(f"  {'Cells':30s} {adata.shape[0]:>12,} {adata_small_filtered.shape[0]:>12,} {adata_medium_filtered.shape[0]:>12,}")
print(f"  {'Genes':30s} {adata.shape[1]:>12,} {adata_small_filtered.shape[1]:>12,} {adata_medium_filtered.shape[1]:>12,}")
print(f"  {'Batches/Donors':30s} {n_batches:>12} {adata_small_filtered.obs[batch_key].nunique():>12} {adata_medium_filtered.obs[batch_key].nunique():>12}")
if cell_type_key:
    print(f"  {'Cell types':30s} {adata.obs[cell_type_key].nunique():>12} {adata_small_filtered.obs[cell_type_key].nunique():>12} {adata_medium_filtered.obs[cell_type_key].nunique():>12}")

print(f"\n🔑 Key metadata columns:")
print(f"  Batch/Donor: '{batch_key}'")
if cell_type_key:
    print(f"  Cell type: '{cell_type_key}'")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)

print(f"\n💡 Recommended workflow:")
print(f"\n1. Quick test with SMALL subset (~2-5 minutes):")
print(f"   python run_bonemarrowmap_subset_analysis.py --subset small")

print(f"\n2. Full training with MEDIUM subset (~20-60 minutes):")
print(f"   python run_bonemarrowmap_subset_analysis.py --subset medium")

print(f"\n3. Visualize results:")
print(f"   python visualize_bonemarrowmap_subset.py --subset medium")

print("\n" + "=" * 80)
