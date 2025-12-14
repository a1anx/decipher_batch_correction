"""
Run batch-corrected Decipher on BoneMarrowMap dataset

This script:
1. Loads the preprocessed 5k gene dataset
2. Auto-detects batch/donor information
3. Trains the batch-corrected Decipher model
4. Extracts embeddings and attention weights
5. Saves results for visualization

Adapted from run_batch_corrected_on_data_2.py for real data
"""

import scanpy as sc
import numpy as np
import sys
import os

# Add decipher-batch-correction to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../decipher-batch-correction/decipher-bc'))

from decipher_batch_corrected import DecipherBatchCorrectedConfig
from train_batch_corrected import train_batch_corrected_decipher, evaluate_batch_correction

print("=" * 80)
print("BATCH-CORRECTED DECIPHER - Training on BoneMarrowMap")
print("=" * 80)

# Step 1: Load preprocessed data
print("\n[1/5] Loading preprocessed data...")
input_file = "BoneMarrowMap_5k_genes.h5ad"

if not os.path.exists(input_file):
    print(f"\n❌ ERROR: Preprocessed file not found: {input_file}")
    print(f"   Please run: python preprocess_bonemarrowmap.py")
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

# Configure for large dataset (263k cells)
config = DecipherBatchCorrectedConfig(
    # Latent dimensions
    dim_z=10,                       # Match original Decipher
    dim_v=2,                        # Match original Decipher (for trajectory)

    # Batch correction parameters
    n_batches=None,                 # Will be set from data
    batch_emb_dim=64,               # Larger for more batches
    decoder_hidden_dims=[128, 256], # Deeper network for complex data
    n_attention_heads=4,            # Multi-head attention
    combination_mode="concat",      # Concatenate z and batch effect

    # Training parameters (optimized for large dataset)
    learning_rate=5e-3,
    batch_size=512,                 # Larger batch for 263k cells
    n_epochs=200,                   # More epochs for real data
    early_stopping_patience=20,     # More patience for large dataset
    val_frac=0.05                   # 5% validation (still 13k cells)
)

config.initialize_from_adata(adata, batch_key=batch_key)

print(f"✓ Configuration initialized:")
print(f"  Architecture:")
print(f"    - Latent z dimension: {config.dim_z}")
print(f"    - Component v dimension: {config.dim_v}")
print(f"    - Batch embedding dimension: {config.batch_emb_dim}")
print(f"    - Decoder hidden layers: {config.decoder_hidden_dims}")
print(f"    - Attention heads: {config.n_attention_heads}")
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
sec_per_batch = 0.5 if device == 'cuda' else 2.0  # Rough estimate
est_time_per_epoch = (batches_per_epoch * sec_per_batch) / 60  # minutes

print(f"\n⏱ Estimated training time:")
print(f"  Time per epoch: ~{est_time_per_epoch:.1f} minutes")
print(f"  Total time (if {config.n_epochs} epochs): ~{est_time_per_epoch * config.n_epochs / 60:.1f} hours")
print(f"  (Early stopping will likely finish sooner)")

# Step 4: Train model
print("\n[4/5] Training model...")
print(f"  Training on {device.upper()}")
print(f"  This will take several hours for {adata.shape[0]:,} cells...")
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
output_file = "bonemarrowmap_batch_corrected.h5ad"
adata.write(output_file)
print(f"✓ Results saved to: {output_file}")

# Save loss curves
print("\nSaving loss curves...")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
epochs = range(1, len(losses['train_losses']) + 1)
ax.plot(epochs, losses['train_losses'], label='Training Loss', linewidth=2)
ax.plot(epochs, losses['val_losses'], label='Validation Loss', linewidth=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bonemarrowmap_loss_curves.png', dpi=300, bbox_inches='tight')
print("✓ Loss curves saved to: bonemarrowmap_loss_curves.png")
plt.close()

# Print summary statistics
print("\n" + "=" * 80)
print("TRAINING SUMMARY")
print("=" * 80)

print("\n📊 Model Performance:")
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
print("OUTPUTS SAVED")
print("=" * 80)

print(f"\n1. {output_file}")
print(f"   - AnnData object with batch-corrected embeddings")
print(f"   - Contains:")
print(f"     • adata.obsm['X_decipher_batch_corrected_z']: Latent representation (10D)")
print(f"     • adata.obsm['X_decipher_batch_corrected_v']: Component representation (2D)")
print(f"     • adata.obsm['X_decipher_batch_corrected_umap']: UMAP embedding")
print(f"     • adata.obs['batch_attention_strength']: Attention per cell")

print(f"\n2. bonemarrowmap_loss_curves.png")
print(f"   - Training and validation loss curves")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)

print("\n1. Visualize batch correction results:")
print("   python visualize_bonemarrowmap.py")

print("\n2. Analyze attention patterns:")
print("   - High attention = strong batch effect on that cell")
print("   - Low attention = minimal batch effect")

print("\n3. Compare with standard methods:")
print("   - PCA, UMAP on raw data")
print("   - scVI, Harmony, etc.")

print("\n" + "=" * 80)
print("✓ TRAINING COMPLETE!")
print("=" * 80)
