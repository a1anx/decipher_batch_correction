"""
Run batch-corrected Decipher with beta=1.0 AND 2 attention heads - GCP Version

This script is optimized for running on Google Cloud Platform with GPU.

Input: BoneMarrowMap_Annotated_Dataset_expandedFeatures.h5ad (3.5 GB, ~263k cells, 34k genes)
Note: Will preprocess to 5k genes on GCP (requires high-memory instance)

Expected runtime on GCP:
- Preprocessing: ~10-15 min (n1-highmem-16, 64 GB RAM)
- Training: ~45-60 min (with T4 GPU, 150 epochs)
- Total: ~1-1.5 hours
- Cost: ~$2-3

Usage:
    python run_bonemarrowmap_full_gcp.py [--epochs N] [--skip-preprocessing]
"""

import scanpy as sc
import numpy as np
import sys
import os
import argparse
import time

# Add decipher-batch-correction to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../decipher-batch-correction/decipher-bc'))

from decipher_batch_corrected import DecipherBatchCorrectedConfig
from train_batch_corrected import train_batch_corrected_decipher, evaluate_batch_correction

# Parse arguments
parser = argparse.ArgumentParser(description='Train batch-corrected Decipher on FULL BoneMarrowMap dataset (GCP)')
parser.add_argument('--epochs', type=int, default=150,
                   help='Number of epochs (default: 150)')
parser.add_argument('--skip-preprocessing', action='store_true',
                   help='Skip preprocessing step (use if BoneMarrowMap_full_5k_genes.h5ad already exists)')
args = parser.parse_args()

# File paths
input_file_full = "BoneMarrowMap_Annotated_Dataset_expandedFeatures.h5ad"
preprocessed_file = "BoneMarrowMap_full_5k_genes.h5ad"
output_dir = "Full_Beta_1.0_AttentionHeads_2_Training"
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("BATCH-CORRECTED DECIPHER (beta=1.0 + 2 Heads) - GCP FULL DATASET")
print("=" * 80)

# Step 0: Preprocess if needed
if args.skip_preprocessing:
    print("\n[0/6] Skipping preprocessing (using existing preprocessed file)")
    if not os.path.exists(preprocessed_file):
        print(f"❌ ERROR: Preprocessed file not found: {preprocessed_file}")
        print("   Run without --skip-preprocessing flag first")
        sys.exit(1)
    input_file = preprocessed_file
else:
    print("\n[0/6] Preprocessing full dataset to 5k genes...")
    print("   This requires ~50-60 GB RAM (make sure you're on n1-highmem-16)")

    if not os.path.exists(input_file_full):
        print(f"❌ ERROR: Full dataset not found: {input_file_full}")
        sys.exit(1)

    start_time = time.time()

    # Load in backed mode
    print(f"   Loading {input_file_full}...")
    adata_backed = sc.read_h5ad(input_file_full, backed='r')
    print(f"   Loaded: {adata_backed.shape[0]:,} cells × {adata_backed.shape[1]:,} genes")

    # Load into memory (this is where we need the RAM)
    print("   Loading into memory (requires ~50 GB RAM)...")
    adata = adata_backed.to_memory()
    print(f"   ✓ In memory: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")

    # Detect batch column
    donor_cols = [col for col in adata.obs.columns
                  if 'donor' in col.lower() or 'batch' in col.lower()]
    batch_key = donor_cols[0] if donor_cols else 'Donor'
    print(f"   Batch column: {batch_key}")

    # Normalize and select HVGs
    print("   Normalizing and selecting top 5k HVGs...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=5000,
        batch_key=batch_key,
        flavor='seurat_v3'
    )

    # Subset to HVGs
    adata_filtered = adata[:, adata.var['highly_variable']].copy()
    print(f"   ✓ Filtered: {adata_filtered.shape[0]:,} cells × {adata_filtered.shape[1]:,} genes")

    # Save preprocessed file
    print(f"   Saving to {preprocessed_file}...")
    adata_filtered.write_h5ad(preprocessed_file)

    elapsed = time.time() - start_time
    print(f"   ✓ Preprocessing complete in {elapsed/60:.1f} minutes")

    input_file = preprocessed_file
    del adata_backed, adata, adata_filtered  # Free memory

# Step 1: Load preprocessed data
print(f"\n[1/5] Loading preprocessed data...")
print(f"   Loading from: {input_file}")

adata = sc.read_h5ad(input_file)
print(f"✓ Loaded data: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")

# Step 2: Auto-detect batch/donor column
print("\n[2/5] Detecting batch/donor column...")

donor_cols = [col for col in adata.obs.columns
              if 'donor' in col.lower() or 'batch' in col.lower()]

if donor_cols:
    batch_key = donor_cols[0]
    print(f"✓ Auto-detected batch column: '{batch_key}'")
else:
    print("❌ ERROR: No batch/donor column found!")
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

# Detect cell type column
cell_type_cols = [col for col in adata.obs.columns
                  if 'type' in col.lower() or 'celltype' in col.lower()]

if cell_type_cols:
    cell_type_key = cell_type_cols[0]
    print(f"\n✓ Detected cell type column: '{cell_type_key}'")
    n_types = adata.obs[cell_type_key].nunique()
    print(f"  Number of cell types: {n_types}")

# Step 3: Configure model
print("\n[3/5] Configuring model for FULL dataset...")

# Detect device
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
    print(f"  Device: CPU")

# Configure model
config = DecipherBatchCorrectedConfig(
    # Latent dimensions
    dim_z=10,
    dim_v=2,

    # Batch correction parameters
    n_batches=None,
    batch_emb_dim=64,
    decoder_hidden_dims=[],
    n_attention_heads=2,
    combination_mode="concat",

    # Training parameters
    beta=1.0,
    learning_rate=5e-3,
    batch_size=512,
    n_epochs=args.epochs,
    early_stopping_patience=15,
    val_frac=0.1
)

config.initialize_from_adata(adata, batch_key=batch_key)

print(f"✓ Configuration initialized:")
print(f"  Architecture:")
print(f"    - Latent z dimension: {config.dim_z}")
print(f"    - Component v dimension: {config.dim_v}")
print(f"    - Batch embedding dimension: {config.batch_emb_dim}")
print(f"    - Attention heads: {config.n_attention_heads}")
print(f"    - Beta (KL weight): {config.beta}")
print(f"  Dataset:")
print(f"    - Cells: {config.n_cells:,}")
print(f"    - Genes: {config.dim_genes:,}")
print(f"    - Batches: {config.n_batches}")
print(f"  Training:")
print(f"    - Batch size: {config.batch_size}")
print(f"    - Max epochs: {config.n_epochs}")

# Estimate training time
batches_per_epoch = config.n_cells // config.batch_size
sec_per_batch = 0.3 if device == 'cuda' else 1.5
est_time_per_epoch = (batches_per_epoch * sec_per_batch) / 60

print(f"\n⏱ Estimated training time:")
print(f"  Time per epoch: ~{est_time_per_epoch:.1f} minutes")
print(f"  Total time (if {config.n_epochs} epochs): ~{est_time_per_epoch * config.n_epochs:.1f} minutes (~{est_time_per_epoch * config.n_epochs / 60:.1f} hours)")

# Step 4: Train model
print("\n[4/5] Training model on FULL dataset...")
print(f"  Training on {device.upper()}")
print(f"  Using beta={config.beta} with {config.n_attention_heads} attention heads")
print()

start_time = time.time()

model, losses = train_batch_corrected_decipher(
    adata,
    batch_key=batch_key,
    config=config,
    device=device
)

training_time = time.time() - start_time

print(f"\n✓ Training complete in {training_time/60:.1f} minutes!")
print(f"  Epochs trained: {len(losses['train_losses'])}")
print(f"  Final training loss: {losses['train_losses'][-1]:.4f}")
print(f"  Final validation loss: {losses['val_losses'][-1]:.4f}")
print(f"  Best validation loss: {min(losses['val_losses']):.4f} (epoch {np.argmin(losses['val_losses']) + 1})")

# Step 5: Extract embeddings
print("\n[5/5] Extracting embeddings and attention weights...")

results = evaluate_batch_correction(model, adata, batch_key=batch_key, device=device)

adata.obsm['X_decipher_batch_corrected_z'] = results['z']
adata.obsm['X_decipher_batch_corrected_v'] = results['v']
adata.obs['batch_attention_strength'] = results['attention_weights'].mean(axis=(1, 2, 3))

print(f"✓ Embeddings extracted:")
print(f"  - X_decipher_batch_corrected_z: {results['z'].shape}")
print(f"  - X_decipher_batch_corrected_v: {results['v'].shape}")

# Compute UMAP
print("\nComputing UMAP embeddings...")
sc.pp.neighbors(adata, use_rep='X_decipher_batch_corrected_z', n_neighbors=15, key_added='batch_corrected')
sc.tl.umap(adata, neighbors_key='batch_corrected')
adata.obsm['X_decipher_batch_corrected_umap'] = adata.obsm['X_umap'].copy()
print("✓ UMAP computed")

# Save results
print("\nSaving results...")
output_file = f'bonemarrowmap_full_{args.epochs}epochs_beta1_attnheads2.h5ad'
output_path = os.path.join(output_dir, output_file)
adata.write(output_path)
print(f"✓ Results saved to: {output_path}")

# Save loss curves
print("\nSaving loss curves...")
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for GCP
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
epochs = range(1, len(losses['train_losses']) + 1)
ax.plot(epochs, losses['train_losses'], label='Training Loss', linewidth=2)
ax.plot(epochs, losses['val_losses'], label='Validation Loss', linewidth=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title(f'Training Loss - FULL Dataset (263k cells)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()

loss_file = f'bonemarrowmap_full_{args.epochs}epochs_loss_curves.png'
loss_file_path = os.path.join(output_dir, loss_file)
plt.savefig(loss_file_path, dpi=300, bbox_inches='tight')
print(f"✓ Loss curves saved to: {loss_file_path}")

# Print summary
print("\n" + "=" * 80)
print("TRAINING SUMMARY - FULL DATASET")
print("=" * 80)

print(f"\n📊 Model Performance:")
print(f"  Training time:   {training_time/60:.1f} minutes")
print(f"  Training loss:   {losses['train_losses'][-1]:.4f}")
print(f"  Validation loss: {losses['val_losses'][-1]:.4f}")
print(f"  Epochs trained:  {len(losses['train_losses'])}/{config.n_epochs}")

print("\n📈 Batch Attention Statistics:")
print(f"  Mean attention: {adata.obs['batch_attention_strength'].mean():.4f}")
print(f"  Std attention:  {adata.obs['batch_attention_strength'].std():.4f}")

print("\n🔍 Attention by batch (top 15):")
batch_attention = adata.obs.groupby(batch_key)['batch_attention_strength'].mean().sort_values(ascending=False)
for i, (batch_id, attn) in enumerate(batch_attention.head(15).items(), 1):
    n_cells = (adata.obs[batch_key] == batch_id).sum()
    print(f"  {i:2d}. {str(batch_id):30s}: {attn:.4f} ({n_cells:,} cells)")

print("\n" + "=" * 80)
print("✓ COMPLETE! Download the following files:")
print("=" * 80)
print(f"\n1. {output_path}")
print(f"2. {loss_file_path}")

print("\n" + "=" * 80)
