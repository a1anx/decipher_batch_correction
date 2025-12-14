"""
Train Regular Decipher (no batch correction) on Erythroid lineage subset.

This serves as the baseline to compare against batch-corrected models.
We expect:
- V-space to show developmental trajectory (early → late erythroblasts)
- But potentially batch effects if donors cluster separately
"""

import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import argparse

# Add decipher-main to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../decipher-main'))

from decipher.tools.decipher import decipher_train
from decipher.tools._decipher import DecipherConfig

parser = argparse.ArgumentParser(description='Train Regular Decipher on Erythroid subset')
parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
parser.add_argument('--output-suffix', type=str, default='', help='Optional output suffix')
args = parser.parse_args()

print("="*80)
print("REGULAR DECIPHER - ERYTHROID LINEAGE")
print("="*80)

# Load erythroid subset
print("\n1. Loading erythroid subset...")
adata = sc.read_h5ad("BoneMarrowMap_Erythroid_5kgenes.h5ad")

# Decipher requires integer counts - use counts layer if available
if 'counts' in adata.layers:
    print("   Using raw counts from 'counts' layer")
    adata.X = adata.layers['counts'].copy()
else:
    print("   WARNING: No counts layer found, using X (may be normalized)")

print(f"   Cells: {adata.n_obs:,}")
print(f"   Genes: {adata.n_vars:,}")
print(f"   Cell types: {adata.obs['CellType'].nunique()}")
print(f"   Donors: {adata.obs['Donor'].nunique()}")

# Configure Decipher (no batch correction)
print("\n2. Configuring Regular Decipher...")
config = DecipherConfig(
    # Latent space dimensions
    dim_z=10,                      # Z-space (high-dimensional)
    dim_v=2,                       # V-space (2D for visualization)

    # Architecture
    layers_v_to_z=(64,),           # Decoder v->z: single hidden layer
    layers_z_to_x=tuple(),         # Decoder z->x: single linear layer (no hidden layers)

    # Training parameters
    learning_rate=5e-3,
    batch_size=128,
    n_epochs=args.epochs,
    val_frac=0.1,
    early_stopping_patience=15,

    # KL weighting
    beta=0.1,                      # Standard KL weight for regular Decipher

    # Other
    seed=42
)

print("\nConfiguration:")
print(f"   dim_z: {config.dim_z}")
print(f"   dim_v: {config.dim_v}")
print(f"   beta: {config.beta}")
print(f"   learning_rate: {config.learning_rate}")
print(f"   batch_size: {config.batch_size}")
print(f"   n_epochs: {config.n_epochs}")

# Train model
print("\n3. Training Regular Decipher...")
print("-" * 80)
decipher_train(
    adata,
    decipher_config=config,
    device='cpu'
)
print("-" * 80)

# adata is modified in-place
adata_trained = adata

# Extract training history from adata (if available)
history = {
    'train_loss': adata_trained.uns.get('decipher_train_loss', [0]),
    'val_loss': adata_trained.uns.get('decipher_val_loss', [0])
}

print(f"\n4. Training completed!")
if len(history['train_loss']) > 1:
    print(f"   Final epoch: {len(history['train_loss'])}")
    print(f"   Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"   Final val loss: {history['val_loss'][-1]:.4f}")
else:
    print(f"   Training history not available in adata.uns")

# Compute UMAP for comparison
print("\n5. Computing UMAP on V-space...")
sc.pp.neighbors(adata_trained, use_rep='decipher_v', n_neighbors=15)
sc.tl.umap(adata_trained)

# Save results
output_file = f"erythroid_regular_decipher{args.output_suffix}.h5ad"
print(f"\n6. Saving results...")
adata_trained.write(output_file)
print(f"   Saved to: {output_file}")

# Create visualization
print("\n7. Creating visualizations...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: V-space by Cell Type
ax = axes[0, 0]
cell_types = adata_trained.obs['CellType'].astype('category')
colors = plt.cm.viridis(np.linspace(0, 1, len(cell_types.cat.categories)))
color_dict = dict(zip(cell_types.cat.categories, colors))
for ct in cell_types.cat.categories:
    mask = adata_trained.obs['CellType'] == ct
    v_coords = adata_trained.obsm['decipher_v'][mask]
    ax.scatter(v_coords[:, 0], v_coords[:, 1], c=[color_dict[ct]],
               label=ct, alpha=0.6, s=20)
ax.set_xlabel('V1', fontsize=11)
ax.set_ylabel('V2', fontsize=11)
ax.set_title('V-space: Cell Type (Developmental Stage)', fontsize=12, fontweight='bold')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax.grid(alpha=0.3)

# Plot 2: V-space by Donor
ax = axes[0, 1]
donors = adata_trained.obs['Donor'].astype('category')
# Use fewer colors for donors (too many to show all)
donor_colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(donors.cat.categories))))
for i, donor in enumerate(donors.cat.categories[:20]):  # Show first 20 donors
    mask = adata_trained.obs['Donor'] == donor
    v_coords = adata_trained.obsm['decipher_v'][mask]
    ax.scatter(v_coords[:, 0], v_coords[:, 1], c=[donor_colors[i]],
               alpha=0.4, s=10)
ax.set_xlabel('V1', fontsize=11)
ax.set_ylabel('V2', fontsize=11)
ax.set_title('V-space: Donor (Batch)', fontsize=12, fontweight='bold')
ax.text(0.02, 0.98, 'Showing first 20 donors', transform=ax.transAxes,
        fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round',
        facecolor='wheat', alpha=0.5))
ax.grid(alpha=0.3)

# Plot 3: UMAP by Cell Type
ax = axes[0, 2]
for ct in cell_types.cat.categories:
    mask = adata_trained.obs['CellType'] == ct
    umap_coords = adata_trained.obsm['X_umap'][mask]
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
z_coords = adata_trained.obsm['decipher_z']
z_var = np.var(z_coords, axis=0)
ax.bar(range(1, len(z_var) + 1), z_var, color='steelblue', edgecolor='black')
ax.set_xlabel('Z Dimension', fontsize=11)
ax.set_ylabel('Variance', fontsize=11)
ax.set_title('Z-space Variance per Dimension', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Plot 6: Summary statistics
ax = axes[1, 2]
ax.axis('off')
summary_text = f"""
REGULAR DECIPHER - ERYTHROID LINEAGE

DATASET:
  Cells: {adata_trained.n_obs:,}
  Genes: {adata_trained.n_vars:,}
  Cell Types: {adata_trained.obs['CellType'].nunique()}
  Donors: {adata_trained.obs['Donor'].nunique()}

MODEL CONFIG:
  dim_z: {config.dim_z}
  dim_v: {config.dim_v}
  beta: {config.beta}
  learning_rate: {config.learning_rate}
  batch_size: {config.batch_size}

TRAINING:
  Epochs: {len(history['train_loss'])}
  Final Train Loss: {history['train_loss'][-1]:.4f}
  Final Val Loss: {history['val_loss'][-1]:.4f}

OUTPUTS:
  ✓ Z-space embeddings ({config.dim_z}D)
  ✓ V-space embeddings (2D)
  ✓ UMAP coordinates

EXPECTED RESULTS:
  - V-space shows developmental trajectory
  - Potential batch effects (donor clustering)
  - Baseline for batch correction comparison
"""
ax.text(0.1, 0.5, summary_text, fontsize=9, family='monospace',
        verticalalignment='center', bbox=dict(boxstyle='round',
        facecolor='lightblue', alpha=0.3))

plt.tight_layout()
plot_file = f"erythroid_regular_decipher{args.output_suffix}_results.png"
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
print(f"   Saved to: {plot_file}")

print("\n" + "="*80)
print("✓ REGULAR DECIPHER TRAINING COMPLETE!")
print("="*80)
print(f"\nOutputs:")
print(f"  1. {output_file}")
print(f"  2. {plot_file}")
print(f"\nNext steps:")
print(f"  - Train batch-corrected model (Beta=1.0, 2 heads)")
print(f"  - Compare V-space structure")
print(f"  - Analyze marker gene expression along trajectory")
