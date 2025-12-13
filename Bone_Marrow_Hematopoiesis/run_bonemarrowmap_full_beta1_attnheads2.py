"""
Run batch-corrected Decipher with beta=1.0 AND 2 attention heads - FULL DATASET

This script runs on the complete BoneMarrowMap dataset (263k cells) with:
- beta=1.0 (stronger KL regularization for structured latent spaces)
- n_attention_heads=2 (reduced attention complexity)
- batch_emb_dim=64
- decoder_hidden_dims=[] (simplified decoder)
- combination_mode="concat" (for stability)

Input: BoneMarrowMap_Annotated_Dataset_expandedFeatures.h5ad (3.5 GB, ~263k cells)
Output: Full_Beta_1.0_AttentionHeads_2_Training/ directory

Expected runtime:
- CPU: ~2-4 hours (150 epochs with early stopping)
- GPU: ~1 hour

Usage:
    python run_bonemarrowmap_full_beta1_attnheads2.py [--epochs N]
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
parser = argparse.ArgumentParser(description='Train batch-corrected Decipher on FULL BoneMarrowMap dataset')
parser.add_argument('--epochs', type=int, default=150,
                   help='Number of epochs (default: 150)')
args = parser.parse_args()

# Configuration for FULL dataset
input_file = "BoneMarrowMap_Annotated_Dataset_expandedFeatures.h5ad"
n_epochs = args.epochs
batch_size = 512  # Larger batch size for full dataset
output_dir = "Full_Beta_1.0_AttentionHeads_2_Training"
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("BATCH-CORRECTED DECIPHER (beta=1.0 + 2 Attention Heads) - FULL DATASET")
print("=" * 80)

# Step 1: Load FULL dataset
print("\n[1/5] Loading FULL BoneMarrowMap dataset...")
print(f"  ⚠️  WARNING: This is the FULL 3.5 GB dataset with ~263k cells")
print(f"  ⚠️  This will require ~8-16 GB RAM and take 2-4 hours on CPU")

if not os.path.exists(input_file):
    print(f"\n❌ ERROR: Full dataset not found: {input_file}")
    print(f"   Expected location: Bone_Marrow_Hematopoiesis/{input_file}")
    sys.exit(1)

adata = sc.read_h5ad(input_file)
print(f"✓ Loaded data: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
print(f"  File size: ~3.5 GB")

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
print("\n[3/5] Configuring model for FULL dataset...")

# Automatically detect device
try:
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device.upper()}")
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
except:
    device = 'cpu'
    print(f"  Device: CPU (no GPU available)")

# Configure for FULL dataset with beta=1.0 AND 2 attention heads
config = DecipherBatchCorrectedConfig(
    # Latent dimensions
    dim_z=10,                       # Match original Decipher
    dim_v=2,                        # Match original Decipher (for trajectory)

    # Batch correction parameters
    n_batches=None,                 # Will be set from data
    batch_emb_dim=64,               # Keep at 64
    decoder_hidden_dims=[],         # Single linear layer (like regular Decipher)
    n_attention_heads=2,            # REDUCED from 4 to 2 (simpler attention)
    combination_mode="concat",      # CONCAT mode (stable)

    # Training parameters (beta=1.0 for stronger regularization)
    beta=1.0,                       # INCREASED from 0.1 to 1.0
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
print(f"    - Attention heads: {config.n_attention_heads} (REDUCED from 4)")
print(f"    - Combination mode: {config.combination_mode} (CONCAT)")
print(f"    - Beta (KL weight): {config.beta} (INCREASED from 0.1)")
print(f"  Dataset:")
print(f"    - Cells: {config.n_cells:,}")
print(f"    - Genes: {config.dim_genes:,}")
print(f"    - Batches: {config.n_batches}")
print(f"  Training:")
print(f"    - Batch size: {config.batch_size}")
print(f"    - Max epochs: {config.n_epochs}")
print(f"    - Early stopping patience: {config.early_stopping_patience}")

# Estimate training time for FULL dataset
cells_per_epoch = config.n_cells
batches_per_epoch = cells_per_epoch // config.batch_size
sec_per_batch = 0.5 if device == 'cuda' else 2.0  # Full dataset is slower per batch
est_time_per_epoch = (batches_per_epoch * sec_per_batch) / 60  # minutes

print(f"\n⏱ Estimated training time for FULL dataset:")
print(f"  Batches per epoch: {batches_per_epoch:,}")
print(f"  Time per epoch: ~{est_time_per_epoch:.1f} minutes")
print(f"  Total time (if {config.n_epochs} epochs): ~{est_time_per_epoch * config.n_epochs:.1f} minutes (~{est_time_per_epoch * config.n_epochs / 60:.1f} hours)")
print(f"  (Early stopping will likely finish sooner)")
print(f"\n  ⚠️  EXPECT: {'~1 hour on GPU' if device == 'cuda' else '~2-4 hours on CPU'}")

# Confirm before starting (skip if not interactive)
import sys
if sys.stdin.isatty():
    print("\n" + "=" * 80)
    print("READY TO START TRAINING")
    print("=" * 80)
    print(f"\nThis will train on {config.n_cells:,} cells for up to {config.n_epochs} epochs")
    print(f"Estimated time: {'~1 hour (GPU)' if device == 'cuda' else '~2-4 hours (CPU)'}")
    print(f"\nPress Ctrl+C to cancel, or Enter to start training...")
    try:
        input()
    except KeyboardInterrupt:
        print("\n\n❌ Training cancelled by user")
        sys.exit(0)
else:
    print("\n" + "=" * 80)
    print("STARTING TRAINING (non-interactive mode)")
    print("=" * 80)

# Step 4: Train model
print("\n[4/5] Training model on FULL dataset...")
print(f"  Training on {device.upper()}")
print(f"  Using beta={config.beta} with {config.n_attention_heads} attention heads")
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
print("  ⚠️  This may take several minutes for 263k cells...")
sc.pp.neighbors(adata, use_rep='X_decipher_batch_corrected_z', n_neighbors=15, key_added='batch_corrected')
sc.tl.umap(adata, neighbors_key='batch_corrected')
adata.obsm['X_decipher_batch_corrected_umap'] = adata.obsm['X_umap'].copy()
print("✓ UMAP computed on batch-corrected embeddings")

# Save results
print("\nSaving results...")
output_file = f'bonemarrowmap_full_{n_epochs}epochs_beta1_attnheads2.h5ad'
output_path = os.path.join(output_dir, output_file)
print(f"  Writing {adata.shape[0]:,} cells to disk (this may take a few minutes)...")
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
ax.set_title(f'Training and Validation Loss - beta=1.0 + 2 Heads (FULL 263k cells)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()

loss_file = f'bonemarrowmap_full_{n_epochs}epochs_beta1_attnheads2_loss_curves.png'
loss_file_path = os.path.join(output_dir, loss_file)
plt.savefig(loss_file_path, dpi=300, bbox_inches='tight')
print(f"✓ Loss curves saved to: {loss_file_path}")
plt.close()

# Print summary statistics
print("\n" + "=" * 80)
print("TRAINING SUMMARY (beta=1.0 + 2 Attention Heads - FULL DATASET)")
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

print("\n🔍 Attention by batch (top 15):")
batch_attention = adata.obs.groupby(batch_key)['batch_attention_strength'].mean().sort_values(ascending=False)
for i, (batch_id, attn) in enumerate(batch_attention.head(15).items(), 1):
    n_cells = (adata.obs[batch_key] == batch_id).sum()
    print(f"  {i:2d}. {str(batch_id):30s}: {attn:.4f} ({n_cells:,} cells)")

if len(batch_attention) > 15:
    print(f"  ... and {len(batch_attention) - 15} more batches")

print("\n" + "=" * 80)
print("OUTPUTS SAVED TO Full_Beta_1.0_AttentionHeads_2_Training/")
print("=" * 80)

print(f"\n1. {output_file}")
print(f"   - AnnData object with FULL dataset (263k cells) batch-corrected embeddings")
print(f"   - Contains:")
print(f"     • adata.obsm['X_decipher_batch_corrected_z']: Latent representation (10D)")
print(f"     • adata.obsm['X_decipher_batch_corrected_v']: Component representation (2D)")
print(f"     • adata.obsm['X_decipher_batch_corrected_umap']: UMAP embedding")
print(f"     • adata.obs['batch_attention_strength']: Attention per cell")

print(f"\n2. {loss_file}")
print(f"   - Training and validation loss curves")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)
print("\n1. Visualize the results:")
print("   - Load the .h5ad file in scanpy/jupyter")
print("   - Plot UMAP colored by batch, cell type, pseudotime")
print("   - Analyze batch attention patterns")

print("\n2. Compare with small subset results:")
print("   - Check if patterns are similar to small dataset")
print("   - Full dataset should show more refined structure")

print("\n3. Run downstream analyses:")
print("   - Differential expression")
print("   - Trajectory inference")
print("   - Cell type annotation refinement")

print("\n" + "=" * 80)
print("✓ TRAINING COMPLETE (FULL DATASET)!")
print("=" * 80)
