"""
Train Simple Concatenation-based Batch-Corrected Decipher on Erythroid lineage.

This script:
1. Loads the erythroid subset
2. Trains concatenation-based batch-corrected Decipher (beta=1.0)
3. Extracts embeddings
4. Saves results to Models directory

The concatenation approach uses concat([z, batch_emb]) → MLP
instead of attention-based batch correction.
"""

import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse

# Add decipher-batch-correction concat module to path
sys.path.insert(0, '/home/alan/Documents/decipher_batch_correction/decipher-batch-correction/decipher-bc-concat')

from decipher_batch_corrected_concat import DecipherBatchCorrectedConcatConfig
from train_batch_corrected_concat import train_batch_corrected_decipher_concat, evaluate_batch_correction_concat

parser = argparse.ArgumentParser(description='Train Concatenation Decipher on Erythroid subset')
parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
parser.add_argument('--beta', type=float, default=1.0, help='KL regularization weight')
parser.add_argument('--output-suffix', type=str, default='', help='Optional output suffix')
args = parser.parse_args()

print("="*80)
print(f"CONCATENATION-BASED BATCH CORRECTION - ERYTHROID LINEAGE (Beta={args.beta})")
print("="*80)

# Load erythroid subset
print("\n1. Loading erythroid subset...")
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
models_dir = os.path.join(base_dir, "Models")

adata = sc.read_h5ad(os.path.join(models_dir, "BoneMarrowMap_Erythroid_5kgenes.h5ad"))
print(f"   Cells: {adata.n_obs:,}")
print(f"   Genes: {adata.n_vars:,}")
print(f"   Cell types: {adata.obs['CellType'].nunique()}")
print(f"   Donors: {adata.obs['Donor'].nunique()}")

# Configure Concatenation-based Batch-Corrected Decipher
print("\n2. Configuring Concatenation-based Batch-Corrected Decipher...")
config = DecipherBatchCorrectedConcatConfig(
    # Latent space dimensions
    dim_z=10,                      # Z-space (high-dimensional)
    dim_v=2,                       # V-space (2D for visualization)

    # Batch correction parameters
    n_batches=None,                # Will be set from data
    batch_emb_dim=64,              # Batch embedding dimension
    decoder_hidden_dims=[],        # Single linear layer (simple decoder)

    # Training parameters
    beta=args.beta,                # KL regularization weight
    learning_rate=5e-3,
    batch_size=128,
    n_epochs=args.epochs,
    early_stopping_patience=15,
)

# Initialize from data
config.initialize_from_adata(adata, batch_key='Donor')

print("\nConfiguration:")
print(f"   dim_z: {config.dim_z}")
print(f"   dim_v: {config.dim_v}")
print(f"   batch_emb_dim: {config.batch_emb_dim}")
print(f"   decoder_hidden_dims: {config.decoder_hidden_dims}")
print(f"   beta: {config.beta}")
print(f"   learning_rate: {config.learning_rate}")
print(f"   batch_size: {config.batch_size}")
print(f"   n_epochs: {config.n_epochs}")
print(f"   Approach: CONCATENATION (no attention mechanism)")

# Train model
print("\n3. Training Concatenation-based Batch-Corrected Decipher...")
print("-" * 80)
model, losses = train_batch_corrected_decipher_concat(
    adata=adata,
    batch_key='Donor',             # Use Donor as batch variable
    config=config,
    device='cpu'
)
print("-" * 80)

# Extract training history
history = {
    'train_loss': losses['train_losses'],
    'val_loss': losses['val_losses']
}

print(f"\n4. Training completed!")
print(f"   Final epoch: {len(history['train_loss'])}")
print(f"   Final train loss: {history['train_loss'][-1]:.4f}")
print(f"   Final val loss: {history['val_loss'][-1]:.4f}")

# Compute embeddings on the full dataset
print("\n5. Computing embeddings...")
results = evaluate_batch_correction_concat(model, adata, batch_key='Donor', device='cpu')

# Store embeddings in adata
adata.obsm['X_decipher_z'] = results['z']
adata.obsm['X_decipher_v'] = results['v']
adata.uns['decipher_train_loss'] = history['train_loss']
adata.uns['decipher_val_loss'] = history['val_loss']
# Convert batch_mapping keys to strings for h5ad compatibility
adata.uns['decipher_batch_mapping'] = {str(k): v for k, v in results['batch_mapping'].items()}

print(f"   Z-space shape: {results['z'].shape}")
print(f"   V-space shape: {results['v'].shape}")

# Compute UMAP for comparison
print("\n6. Computing UMAP on V-space...")
sc.pp.neighbors(adata, use_rep='X_decipher_v', n_neighbors=15)
sc.tl.umap(adata)

# Save results
output_file = os.path.join(models_dir, f"erythroid_concat_beta{args.beta}{args.output_suffix}.h5ad")
print(f"\n7. Saving results...")
adata.write(output_file)
print(f"   Saved to: {output_file}")

# Create visualization
print("\n8. Creating visualizations...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: V-space by Cell Type
ax = axes[0, 0]
cell_types = adata.obs['CellType'].astype('category')
colors = plt.cm.viridis(np.linspace(0, 1, len(cell_types.cat.categories)))
color_dict = dict(zip(cell_types.cat.categories, colors))
for ct in cell_types.cat.categories:
    mask = adata.obs['CellType'] == ct
    v_coords = adata.obsm['X_decipher_v'][mask]
    ax.scatter(v_coords[:, 0], v_coords[:, 1], c=[color_dict[ct]],
               label=ct, alpha=0.6, s=20)
ax.set_xlabel('V1', fontsize=11)
ax.set_ylabel('V2', fontsize=11)
ax.set_title('V-space: Cell Type (Should show trajectory)', fontsize=12, fontweight='bold')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax.grid(alpha=0.3)

# Plot 2: V-space by Donor
ax = axes[0, 1]
donors = adata.obs['Donor'].astype('category')
# Use fewer colors for donors
donor_colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(donors.cat.categories))))
for i, donor in enumerate(donors.cat.categories[:20]):  # Show first 20 donors
    mask = adata.obs['Donor'] == donor
    v_coords = adata.obsm['X_decipher_v'][mask]
    ax.scatter(v_coords[:, 0], v_coords[:, 1], c=[donor_colors[i]],
               alpha=0.4, s=10)
ax.set_xlabel('V1', fontsize=11)
ax.set_ylabel('V2', fontsize=11)
ax.set_title('V-space: Donor (Should be mixed)', fontsize=12, fontweight='bold')
ax.text(0.02, 0.98, 'Showing first 20 donors', transform=ax.transAxes,
        fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round',
        facecolor='wheat', alpha=0.5))
ax.grid(alpha=0.3)

# Plot 3: UMAP by Cell Type
ax = axes[0, 2]
for ct in cell_types.cat.categories:
    mask = adata.obs['CellType'] == ct
    umap_coords = adata.obsm['X_umap'][mask]
    ax.scatter(umap_coords[:, 0], umap_coords[:, 1], c=[color_dict[ct]],
               label=ct, alpha=0.6, s=20)
ax.set_xlabel('UMAP1', fontsize=11)
ax.set_ylabel('UMAP2', fontsize=11)
ax.set_title('UMAP: Cell Type', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

# Plot 4: Training curves
ax = axes[1, 0]
epochs = range(1, len(history['train_loss']) + 1)
ax.plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
ax.plot(epochs, history['val_loss'], label='Validation Loss', linewidth=2)
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Loss', fontsize=11)
ax.set_title('Training Curves', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Plot 5: Z-space variance per dimension
ax = axes[1, 1]
z_coords = adata.obsm['X_decipher_z']
z_var = np.var(z_coords, axis=0)
ax.bar(range(1, len(z_var) + 1), z_var, color='teal', edgecolor='black')
ax.set_xlabel('Z Dimension', fontsize=11)
ax.set_ylabel('Variance', fontsize=11)
ax.set_title('Z-space Variance per Dimension', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Plot 6: Summary statistics
ax = axes[1, 2]
ax.axis('off')
summary_text = f"""
CONCATENATION-BASED BC - ERYTHROID

DATASET:
  Cells: {adata.n_obs:,}
  Genes: {adata.n_vars:,}
  Cell Types: {adata.obs['CellType'].nunique()}
  Donors (Batches): {adata.obs['Donor'].nunique()}

MODEL CONFIG:
  dim_z: {config.dim_z}
  dim_v: {config.dim_v}
  batch_emb_dim: {config.batch_emb_dim}
  decoder_hidden_dims: {config.decoder_hidden_dims}
  beta: {config.beta}
  learning_rate: {config.learning_rate}
  batch_size: {config.batch_size}

ARCHITECTURE:
  concat([z, batch_emb]) → MLP
  (No attention mechanism)

TRAINING:
  Epochs: {len(history['train_loss'])}
  Final Train Loss: {history['train_loss'][-1]:.4f}
  Final Val Loss: {history['val_loss'][-1]:.4f}

OUTPUTS:
  ✓ Z-space embeddings ({config.dim_z}D)
  ✓ V-space embeddings (2D)
  ✓ UMAP coordinates

EXPECTED RESULTS:
  - Preserved developmental trajectory
  - Improved donor mixing
  - Simpler than attention approach
"""
ax.text(0.1, 0.5, summary_text, fontsize=9, family='monospace',
        verticalalignment='center', bbox=dict(boxstyle='round',
        facecolor='lightcyan', alpha=0.3))

plt.tight_layout()
plot_file = os.path.join(base_dir, "Visualizations", f"erythroid_concat_beta{args.beta}{args.output_suffix}_results.png")
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
print(f"   Saved to: {plot_file}")

print("\n" + "="*80)
print("✓ CONCATENATION-BASED BATCH-CORRECTED DECIPHER TRAINING COMPLETE!")
print("="*80)
print(f"\nOutputs:")
print(f"  1. {output_file}")
print(f"  2. {plot_file}")
print(f"\nNext steps:")
print(f"  - Compare with attention-based models")
print(f"  - Create pseudotime visualizations")
print(f"  - Analyze marker gene expression")
