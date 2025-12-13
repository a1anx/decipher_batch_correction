"""
Run batch-corrected Decipher with beta=0.5 - Local Machine Version

This script uses beta=0.5 as a middle ground between default (0.1) and strong (1.0).
Testing intermediate regularization strength.

Differences from regular script:
- beta=0.5 (middle ground between 0.1 and 1.0)
- batch_emb_dim=64 (keeping from best previous run)
- decoder_hidden_dims=[] (keeping simplified decoder)
- n_attention_heads=4 (keeping default)
- combination_mode="concat" (keeping for stability)
- Outputs to Beta_0.5_Training/ folder
"""

import scanpy as sc
import numpy as np
import sys
import os
import argparse

# Add decipher-batch-correction to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../decipher-batch-correction/decipher-bc'))

from decipher_batch_corrected import DecipherBatchCorrectedConfig
from train_batch_corrected import train_batch_corrected_decipher, evaluate_batch_correction

# Parse arguments
parser = argparse.ArgumentParser(description='Train batch-corrected Decipher with beta=0.5')
parser.add_argument('--subset', type=str, default='small', choices=['small', 'medium'],
                   help='Which subset to use: small (20k cells, quick test) or medium (90k cells, full training)')
parser.add_argument('--epochs', type=int, default=None,
                   help='Number of epochs (default: 50 for small, 150 for medium)')
args = parser.parse_args()

# Configuration based on subset
if args.subset == 'small':
    input_file = "BoneMarrowMap_small_20k_5kgenes.h5ad"
    default_epochs = 50
    batch_size = 128
    subset_desc = "SMALL (20k cells)"
else:  # medium
    input_file = "BoneMarrowMap_medium_90k_5kgenes.h5ad"
    default_epochs = 150
    batch_size = 256
    subset_desc = "MEDIUM (90k cells)"

n_epochs = args.epochs if args.epochs is not None else default_epochs

# Set output file name and directory
output_dir = "Beta_0.5_Training"
os.makedirs(output_dir, exist_ok=True)
output_file = f"bonemarrowmap_{args.subset}_{n_epochs}epochs_beta0.5.h5ad"

print("=" * 80)
print(f"BATCH-CORRECTED DECIPHER (beta=0.5) - {subset_desc}")
print("=" * 80)

# Step 1: Load preprocessed data
print("\n[1/5] Loading preprocessed data...")

if not os.path.exists(input_file):
    print(f"\n❌ ERROR: Preprocessed file not found: {input_file}")
    print(f"   Please run: python preprocess_bonemarrowmap_subset.py")
    sys.exit(1)

adata = sc.read_h5ad(input_file)
print(f"✓ Loaded data: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")

# Step 2: Auto-detect batch/donor column
print("\n[2/5] Detecting batch/donor column...")

# Check common batch column names
donor_cols = [col for col in adata.obs.columns
              if 'donor' in col.lower() or 'batch' in col.lower()
              or 'sample' in col.lower() or 'patient' in col.lower()]

if not donor_cols:
    print("❌ ERROR: No batch/donor column found!")
    print(f"   Available columns: {list(adata.obs.columns)}")
    print("\n   Please specify manually:")
    batch_key = input("   Enter batch column name: ").strip()
else:
    batch_key = donor_cols[0]
    print(f"✓ Auto-detected batch column: '{batch_key}'")

# Verify batch column exists
if batch_key not in adata.obs.columns:
    print(f"❌ ERROR: Column '{batch_key}' not found in data!")
    sys.exit(1)

n_batches = len(adata.obs[batch_key].unique())
print(f"  Number of batches: {n_batches}")

# Show batch distribution
print("\n  Batch distribution (top 10):")
batch_counts = adata.obs[batch_key].value_counts().head(10)
for batch_id, count in batch_counts.items():
    print(f"    {str(batch_id):30s}: {count:6,} cells")

if n_batches > 10:
    print(f"    ... and {n_batches - 10} more batches")

# Detect cell type column (optional, for QC)
cell_type_cols = [col for col in adata.obs.columns
                  if 'type' in col.lower() or 'cluster' in col.lower()
                  or 'celltype' in col.lower() or 'annotation' in col.lower()]

if cell_type_cols:
    cell_type_key = cell_type_cols[0]
    print(f"\n✓ Detected cell type column: '{cell_type_key}'")
    n_types = adata.obs[cell_type_key].nunique()
    print(f"  Number of cell types: {n_types}")
else:
    cell_type_key = None
    print("\n⚠ No cell type column detected (optional)")

# Step 3: Configure model
print("\n[3/5] Configuring model...")

# Automatically detect device
try:
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device.upper()}")
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  GPU: {gpu_name}")
except:
    device = 'cpu'
    print(f"  Device: CPU (no GPU available)")

# Configure for subset size with beta=0.5
config = DecipherBatchCorrectedConfig(
    # Latent dimensions
    dim_z=10,                       # Match original Decipher
    dim_v=2,                        # Match original Decipher (for trajectory)

    # Batch correction parameters
    n_batches=None,                 # Will be set from data
    batch_emb_dim=64,               # Keep at 64
    decoder_hidden_dims=[],         # Single linear layer (like regular Decipher)
    n_attention_heads=4,            # Keep at 4
    combination_mode="concat",      # CONCAT mode (stable)

    # Training parameters (beta=0.5 is the key change)
    beta=0.5,                       # MIDDLE GROUND between 0.1 and 1.0
    learning_rate=5e-3,
    batch_size=batch_size,
    n_epochs=n_epochs,
    early_stopping_patience=15,
    val_frac=0.1                    # 10% validation
)

config.initialize_from_adata(adata, batch_key=batch_key)

print(f"✓ Configuration initialized:")
print(f"  Architecture:")
print(f"    - Latent z dimension: {config.dim_z}")
print(f"    - Component v dimension: {config.dim_v}")
print(f"    - Batch embedding dimension: {config.batch_emb_dim}")
print(f"    - Decoder hidden layers: {config.decoder_hidden_dims}")
print(f"    - Attention heads: {config.n_attention_heads}")
print(f"    - Combination mode: {config.combination_mode} (CONCAT)")
print(f"    - Beta (KL weight): {config.beta} (MIDDLE GROUND: 0.1 < 0.5 < 1.0)")
print(f"  Dataset:")
print(f"    - Cells: {config.n_cells:,}")
print(f"    - Genes: {config.dim_genes:,}")
print(f"    - Batches: {config.n_batches}")
print(f"  Training:")
print(f"    - Batch size: {config.batch_size}")
print(f"    - Max epochs: {config.n_epochs}")
print(f"    - Early stopping patience: {config.early_stopping_patience}")

# Estimate training time
cells_per_epoch = config.n_cells
batches_per_epoch = cells_per_epoch // config.batch_size
sec_per_batch = 0.5 if device == 'cuda' else 1.5  # Rough estimate
est_time_per_epoch = (batches_per_epoch * sec_per_batch) / 60  # minutes

print(f"\n⏱ Estimated training time:")
print(f"  Time per epoch: ~{est_time_per_epoch:.1f} minutes")
print(f"  Total time (if {config.n_epochs} epochs): ~{est_time_per_epoch * config.n_epochs:.1f} minutes")
print(f"  (Early stopping will likely finish sooner)")

# Step 4: Train model
print("\n[4/5] Training model...")
print(f"  Training on {device.upper()} with {args.subset.upper()} subset")
print(f"  Using beta={config.beta} (intermediate regularization)")
print(f"  This will take approximately {est_time_per_epoch * config.n_epochs:.0f} minutes")
print(f"  Progress will be shown every epoch")
print()

model, losses = train_batch_corrected_decipher(
    adata,
    batch_key=batch_key,
    config=config,
    device=device
)

print(f"\n✓ Training complete!")
print(f"  Epochs trained: {len(losses['train_losses'])}")
print(f"  Final training loss: {losses['train_losses'][-1]:.4f}")
print(f"  Final validation loss: {losses['val_losses'][-1]:.4f}")
print(f"  Best validation loss: {min(losses['val_losses']):.4f} (epoch {np.argmin(losses['val_losses']) + 1})")

# Step 5: Extract embeddings and attention weights
print("\n[5/5] Extracting embeddings and attention weights...")
print("  Computing latent representations...")

results = evaluate_batch_correction(model, adata, batch_key=batch_key, device=device)

# Add to AnnData
adata.obsm['X_decipher_batch_corrected_z'] = results['z']
adata.obsm['X_decipher_batch_corrected_v'] = results['v']
adata.obs['batch_attention_strength'] = results['attention_weights'].mean(axis=(1, 2, 3))

print(f"✓ Embeddings extracted:")
print(f"  - X_decipher_batch_corrected_z: {results['z'].shape}")
print(f"  - X_decipher_batch_corrected_v: {results['v'].shape}")
print(f"  - batch_attention_strength: {results['attention_weights'].shape[0]:,} cells")

# Compute UMAP for visualization
print("\nComputing UMAP embeddings for visualization...")
sc.pp.neighbors(adata, use_rep='X_decipher_batch_corrected_z', n_neighbors=15, key_added='batch_corrected')
sc.tl.umap(adata, neighbors_key='batch_corrected')
adata.obsm['X_decipher_batch_corrected_umap'] = adata.obsm['X_umap'].copy()
print("✓ UMAP computed on batch-corrected embeddings")

# Save results
print("\nSaving results...")
output_path = os.path.join(output_dir, output_file)
adata.write(output_path)
print(f"✓ Results saved to: {output_path}")

# Save loss curves
print("\nSaving loss curves...")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
epochs = range(1, len(losses['train_losses']) + 1)
ax.plot(epochs, losses['train_losses'], label='Training Loss', linewidth=2)
ax.plot(epochs, losses['val_losses'], label='Validation Loss', linewidth=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title(f'Training and Validation Loss - beta=0.5 ({args.subset.upper()} subset)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()

loss_file = f'bonemarrowmap_{args.subset}_{n_epochs}epochs_beta0.5_loss_curves.png'
loss_file_path = os.path.join(output_dir, loss_file)
plt.savefig(loss_file_path, dpi=300, bbox_inches='tight')
print(f"✓ Loss curves saved to: {loss_file_path}")
plt.close()

# Print summary statistics
print("\n" + "=" * 80)
print("TRAINING SUMMARY (beta=0.5)")
print("=" * 80)

print(f"\n📊 Model Performance:")
print(f"  Training loss:   {losses['train_losses'][-1]:.4f}")
print(f"  Validation loss: {losses['val_losses'][-1]:.4f}")
print(f"  Epochs trained:  {len(losses['train_losses'])}/{config.n_epochs}")

print("\n📈 Batch Attention Statistics:")
print(f"  Mean attention: {adata.obs['batch_attention_strength'].mean():.4f}")
print(f"  Std attention:  {adata.obs['batch_attention_strength'].std():.4f}")
print(f"  Min attention:  {adata.obs['batch_attention_strength'].min():.4f}")
print(f"  Max attention:  {adata.obs['batch_attention_strength'].max():.4f}")

print("\n🔍 Attention by batch (top 10):")
batch_attention = adata.obs.groupby(batch_key)['batch_attention_strength'].mean().sort_values(ascending=False)
for i, (batch_id, attn) in enumerate(batch_attention.head(10).items(), 1):
    n_cells = (adata.obs[batch_key] == batch_id).sum()
    print(f"  {i:2d}. {str(batch_id):30s}: {attn:.4f} ({n_cells:,} cells)")

print("\n" + "=" * 80)
print("OUTPUTS SAVED TO Beta_0.5_Training/")
print("=" * 80)

print(f"\n1. {output_file}")
print(f"   - AnnData object with beta=0.5 batch-corrected embeddings")
print(f"   - Contains:")
print(f"     • adata.obsm['X_decipher_batch_corrected_z']: Latent representation (10D)")
print(f"     • adata.obsm['X_decipher_batch_corrected_v']: Component representation (2D)")
print(f"     • adata.obsm['X_decipher_batch_corrected_umap']: UMAP embedding")
print(f"     • adata.obs['batch_attention_strength']: Attention per cell")

print(f"\n2. {loss_file}")
print(f"   - Training and validation loss curves")

print("\n" + "=" * 80)
print("✓ TRAINING COMPLETE (beta=0.5)!")
print("=" * 80)
