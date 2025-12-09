"""
Preprocessing script for BoneMarrowMap dataset

This script:
1. Loads the full BoneMarrowMap dataset
2. Explores metadata structure (donors, cell types, etc.)
3. Filters to top 5k highly variable genes (HVGs)
4. Saves preprocessed dataset for training

The HVG selection preserves batch effects while reducing computational cost.
"""

import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

print("=" * 80)
print("BONEMARROWMAP PREPROCESSING - Gene Filtering")
print("=" * 80)

# Step 1: Load dataset
print("\n[1/5] Loading BoneMarrowMap dataset...")
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
print("\n[2/5] Exploring metadata structure...")
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
print("\nBatch/Donor distribution:")
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

# Step 3: Load data into memory for gene filtering
print("\n[3/5] Loading data into memory for gene filtering...")
print("   This step loads the full dataset - may take a few minutes...")

# Load into memory
adata_memory = adata.to_memory()
print(f"✓ Loaded into memory: {adata_memory.shape[0]:,} cells × {adata_memory.shape[1]:,} genes")

# Basic QC info
print("\nData matrix information:")
print(f"  Matrix type: {type(adata_memory.X)}")
if hasattr(adata_memory.X, 'nnz'):
    sparsity = 1.0 - (adata_memory.X.nnz / (adata_memory.shape[0] * adata_memory.shape[1]))
    print(f"  Sparsity: {sparsity:.2%}")
    print(f"  Non-zero entries: {adata_memory.X.nnz:,}")

# Step 4: Filter to highly variable genes
print("\n[4/5] Selecting highly variable genes...")
print(f"   Filtering to top 5,000 genes using Seurat v3 method")
print(f"   This preserves batch effects while reducing dimensionality")

# Normalize if not already normalized
if 'log1p' not in adata_memory.uns:
    print("\n   Normalizing counts...")
    sc.pp.normalize_total(adata_memory, target_sum=1e4)
    sc.pp.log1p(adata_memory)

# Select highly variable genes
# Use batch_key to preserve batch-specific variation
print(f"   Computing HVGs with batch correction using '{batch_key}'...")
sc.pp.highly_variable_genes(
    adata_memory,
    n_top_genes=5000,
    batch_key=batch_key,
    flavor='seurat_v3',
    subset=False  # Don't subset yet, just mark
)

n_hvg = adata_memory.var['highly_variable'].sum()
print(f"✓ Identified {n_hvg} highly variable genes")

# Subset to HVGs
adata_filtered = adata_memory[:, adata_memory.var['highly_variable']].copy()
print(f"✓ Filtered dataset: {adata_filtered.shape[0]:,} cells × {adata_filtered.shape[1]:,} genes")

# Calculate size reduction
original_size_mb = (adata_memory.X.data.nbytes if hasattr(adata_memory.X, 'data')
                   else adata_memory.X.nbytes) / (1024**2)
filtered_size_mb = (adata_filtered.X.data.nbytes if hasattr(adata_filtered.X, 'data')
                   else adata_filtered.X.nbytes) / (1024**2)

print(f"\nSize reduction:")
print(f"  Original: {original_size_mb:.1f} MB")
print(f"  Filtered: {filtered_size_mb:.1f} MB")
print(f"  Reduction: {(1 - filtered_size_mb/original_size_mb)*100:.1f}%")

# Step 5: Save filtered dataset
print("\n[5/5] Saving filtered dataset...")
output_file = "BoneMarrowMap_5k_genes.h5ad"
adata_filtered.write_h5ad(output_file)
print(f"✓ Saved to: {output_file}")

# Create summary file
print("\nCreating summary statistics file...")
summary_file = "BoneMarrowMap_preprocessing_summary.txt"
with open(summary_file, 'w') as f:
    f.write("BoneMarrowMap Preprocessing Summary\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"Original dataset: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes\n")
    f.write(f"Filtered dataset: {adata_filtered.shape[0]:,} cells × {adata_filtered.shape[1]:,} genes\n\n")

    f.write(f"Batch column: {batch_key}\n")
    f.write(f"Number of batches: {n_batches}\n\n")

    if cell_type_key:
        f.write(f"Cell type column: {cell_type_key}\n")
        f.write(f"Number of cell types: {adata_filtered.obs[cell_type_key].nunique()}\n\n")

    f.write("Top 10 cell types:\n")
    for cell_type, count in adata_filtered.obs[cell_type_key].value_counts().head(10).items():
        f.write(f"  {cell_type}: {count:,} cells\n")

    f.write(f"\nHVG selection method: Seurat v3 with batch correction\n")
    f.write(f"Number of HVGs: {n_hvg}\n")

    f.write(f"\nFile sizes:\n")
    f.write(f"  Original: {original_size_mb:.1f} MB\n")
    f.write(f"  Filtered: {filtered_size_mb:.1f} MB\n")

print(f"✓ Summary saved to: {summary_file}")

# Print summary
print("\n" + "=" * 80)
print("PREPROCESSING COMPLETE")
print("=" * 80)

print(f"\n✓ Output files created:")
print(f"  1. {output_file}")
print(f"     - Filtered dataset ready for training")
print(f"     - {adata_filtered.shape[0]:,} cells × {adata_filtered.shape[1]:,} genes")
print(f"     - {filtered_size_mb:.1f} MB")

print(f"\n  2. {summary_file}")
print(f"     - Summary statistics")

print(f"\n📊 Dataset summary:")
print(f"  Cells: {adata_filtered.shape[0]:,}")
print(f"  Genes: {adata_filtered.shape[1]:,}")
print(f"  Batches: {n_batches}")
if cell_type_key:
    print(f"  Cell types: {adata_filtered.obs[cell_type_key].nunique()}")

print(f"\n🔑 Key metadata columns:")
print(f"  Batch/Donor: '{batch_key}'")
if cell_type_key:
    print(f"  Cell type: '{cell_type_key}'")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)
print(f"\n1. Train Decipher batch correction model:")
print(f"   python run_bonemarrowmap_analysis.py")
print(f"\n2. Visualize results:")
print(f"   python visualize_bonemarrowmap.py")

print("\n" + "=" * 80)
