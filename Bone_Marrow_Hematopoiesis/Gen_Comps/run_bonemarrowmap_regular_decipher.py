"""
Run regular (non-batch-corrected) Decipher on BoneMarrowMap SMALL subset

This script runs the standard Decipher model WITHOUT batch correction
to compare against the batch-corrected version.

Usage:
    python run_bonemarrowmap_regular_decipher.py
"""

import sys
import os
import scanpy as sc
import numpy as np
import logging

# Add decipher-main to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../decipher-main'))

from decipher.tools.decipher import decipher_train
from decipher.tools._decipher import DecipherConfig

# Set up logging
logging.basicConfig(
    format='%(levelname)s:%(name)s:%(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

print("=" * 80)
print("REGULAR DECIPHER - Training on BoneMarrowMap SMALL Subset")
print("=" * 80)

# Load data
print("\n[1/4] Loading data...")
input_file = "BoneMarrowMap_small_20k_5kgenes.h5ad"
adata = sc.read_h5ad(input_file)
print(f"✓ Loaded data: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")

# Check that we have the counts layer
if 'counts' not in adata.layers:
    print("ERROR: No 'counts' layer found. Please run preprocessing first.")
    sys.exit(1)

# Use raw counts for Decipher (NegativeBinomial requires integers)
print("  Using raw counts from adata.layers['counts']")
adata.X = adata.layers['counts'].copy()

# Get metadata info
batch_cols = [col for col in adata.obs.columns
              if 'donor' in col.lower() or 'batch' in col.lower()]
if batch_cols:
    batch_key = batch_cols[0]
    n_batches = adata.obs[batch_key].nunique()
    print(f"  Batch/Donor column: '{batch_key}' ({n_batches} batches)")

cell_type_cols = [col for col in adata.obs.columns
                  if 'celltype' in col.lower() or 'cell_type' in col.lower()]
if cell_type_cols:
    cell_type_key = cell_type_cols[0]
    n_types = adata.obs[cell_type_key].nunique()
    print(f"  Cell type column: '{cell_type_key}' ({n_types} types)")

# Configure model
print("\n[2/4] Configuring regular Decipher model...")
config = DecipherConfig(
    # Latent dimensions (match batch-corrected version)
    dim_z=10,
    dim_v=2,

    # Architecture
    layers_v_to_z=(64,),      # Decoder v->z: single hidden layer with 64 units
    layers_z_to_x=tuple(),    # Decoder z->x: single linear layer (no hidden layers)

    # Training parameters
    learning_rate=5e-3,
    batch_size=128,
    n_epochs=50,
    val_frac=0.1,
    early_stopping_patience=15,

    # KL weighting
    beta=0.1,  # Match default

    # Other
    seed=42
)

print(f"✓ Configuration:")
print(f"  - Latent z dimension: {config.dim_z}")
print(f"  - Component v dimension: {config.dim_v}")
print(f"  - Batch size: {config.batch_size}")
print(f"  - Max epochs: {config.n_epochs}")
print(f"  - Early stopping patience: {config.early_stopping_patience}")

# Train model
print("\n[3/4] Training regular Decipher model...")
print(f"  This may take several minutes...")
print(f"  NOTE: This is the STANDARD Decipher WITHOUT batch correction")
print()

# Train using the official decipher_train function
decipher_train(
    adata,
    decipher_config=config,
    device='cpu'
)

print("\n✓ Training complete!")

# The decipher_train function automatically adds embeddings to adata.obsm:
#   - decipher_z: latent representation
#   - decipher_v: component representation
# Note: No UMAP is computed by default

# Rename to distinguish from batch-corrected version
print("\n[4/4] Extracting and saving embeddings...")
adata.obsm['X_decipher_regular_z'] = adata.obsm['decipher_z'].copy()
adata.obsm['X_decipher_regular_v'] = adata.obsm['decipher_v'].copy()

# Compute UMAP on the latent z space
print("  Computing UMAP on latent z space...")
sc.pp.neighbors(adata, use_rep='decipher_z', n_neighbors=15, key_added='decipher')
sc.tl.umap(adata, neighbors_key='decipher')
adata.obsm['X_decipher_regular_umap'] = adata.obsm['X_umap'].copy()

# Compute UMAP on raw counts for comparison (standard preprocessing)
print("  Computing standard UMAP on log-normalized data...")
adata_temp = adata.copy()
sc.pp.normalize_total(adata_temp, target_sum=1e4)
sc.pp.log1p(adata_temp)
sc.pp.highly_variable_genes(adata_temp, n_top_genes=2000)
sc.pp.pca(adata_temp, n_comps=50)
sc.pp.neighbors(adata_temp, n_neighbors=15)
sc.tl.umap(adata_temp)

# Copy standard preprocessing results back
if 'X_pca' not in adata.obsm:
    adata.obsm['X_pca'] = adata_temp.obsm['X_pca']
if 'X_umap' not in adata.obsm:
    adata.obsm['X_umap'] = adata_temp.obsm['X_umap']

# Save results
output_file = "bonemarrowmap_small_regular_decipher.h5ad"
adata.write_h5ad(output_file)
print(f"✓ Saved results to: {output_file}")

print("\n" + "=" * 80)
print("OUTPUTS SAVED")
print("=" * 80)
print(f"\n1. {output_file}")
print("   - AnnData object with regular Decipher embeddings")
print("   - Contains:")
print("     • adata.obsm['X_decipher_regular_z']: Latent representation (10D)")
print("     • adata.obsm['X_decipher_regular_v']: Component representation (2D)")
print("     • adata.obsm['X_decipher_regular_umap']: UMAP embedding")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)
print("\n1. Create side-by-side comparison with batch-corrected version:")
print("   python create_decipher_comparison.py")

print("\n" + "=" * 80)
print("✓ TRAINING COMPLETE!")
print("=" * 80)
