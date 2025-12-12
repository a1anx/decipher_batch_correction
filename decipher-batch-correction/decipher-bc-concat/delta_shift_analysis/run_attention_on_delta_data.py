"""
Run attention-based batch-corrected Decipher on delta shift data

This script:
1. Loads the delta shift data
2. Uses 'delta' column as batch information
3. Trains the attention-based batch-corrected Decipher model
4. Extracts embeddings and attention weights
5. Saves results back to the AnnData object

This is the MORE COMPLEX approach using multi-head attention.
"""

import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add decipher-bc directory to path to import modules
decipher_bc_path = '/home/alan/Documents/decipher_batch_correction/decipher-batch-correction/decipher-bc'
if decipher_bc_path not in sys.path:
    sys.path.insert(0, decipher_bc_path)

from decipher_batch_corrected import DecipherBatchCorrectedConfig
from train_batch_corrected import train_batch_corrected_decipher, evaluate_batch_correction

print("=" * 70)
print("ATTENTION-BASED BATCH CORRECTION - Training on Delta Shift Data")
print("=" * 70)
print("This uses multi-head ATTENTION mechanism (more complex)")
print("=" * 70)

# Step 1: Load data
print("\n[1/5] Loading data...")
adata = sc.read_h5ad('/home/alan/Documents/decipher_batch_correction/simulation/archive/oldadata_alpha/adata_combined_2_delta.h5ad')
print(f"✓ Loaded data: {adata.shape[0]} cells, {adata.shape[1]} genes")
print(f"  Batch column: 'delta' with {len(adata.obs['delta'].unique())} conditions")
print(f"  Conditions: {sorted(list(adata.obs['delta'].unique()))}")

# Step 2: Configure model
print("\n[2/5] Configuring model...")
config = DecipherBatchCorrectedConfig(
    # Latent dimensions
    dim_z=10,                       # Match original Decipher
    dim_v=2,                        # Match original Decipher

    # Batch correction parameters
    n_batches=None,                 # Will be set from data
    batch_emb_dim=32,               # Batch embedding dimension (same as concat)
    decoder_hidden_dims=[64, 128],  # Decoder MLP architecture (same as concat)
    n_attention_heads=4,            # Multi-head attention
    combination_mode="concat",      # Concatenate z and batch effect

    # Training parameters
    learning_rate=5e-3,
    batch_size=128,
    n_epochs=100,
    early_stopping_patience=15
)

config.initialize_from_adata(adata, batch_key='delta')
print(f"✓ Configuration initialized:")
print(f"  - Latent z dimension: {config.dim_z}")
print(f"  - Component v dimension: {config.dim_v}")
print(f"  - Number of batches: {config.n_batches}")
print(f"  - Batch embedding dimension: {config.batch_emb_dim}")
print(f"  - Attention heads: {config.n_attention_heads}")
print(f"  - Approach: ATTENTION (more complex than concatenation)")

# Step 3: Train model
print("\n[3/5] Training model...")
print(f"  Training for up to {config.n_epochs} epochs (early stopping: {config.early_stopping_patience})")
print(f"  This may take several minutes...")

model, losses = train_batch_corrected_decipher(
    adata,
    batch_key='delta',
    config=config,
    device='cpu'  # Use CPU
)

print(f"✓ Training complete!")
print(f"  Final training loss: {losses['train_losses'][-1]:.2f}")
print(f"  Final validation loss: {losses['val_losses'][-1]:.2f}")
print(f"  Trained for {len(losses['train_losses'])} epochs")

# Step 4: Extract embeddings and attention weights
print("\n[4/5] Extracting embeddings and attention weights...")
results = evaluate_batch_correction(model, adata, batch_key='delta', device='cpu')

# Add to AnnData
adata.obsm['X_decipher_batch_corrected_z'] = results['z']
adata.obsm['X_decipher_batch_corrected_v'] = results['v']
adata.obs['batch_attention_strength'] = results['attention_weights'].mean(axis=(1, 2, 3))

print(f"✓ Embeddings extracted:")
print(f"  - X_decipher_batch_corrected_z: {results['z'].shape}")
print(f"  - X_decipher_batch_corrected_v: {results['v'].shape}")
print(f"  - batch_attention_strength: {results['attention_weights'].shape[0]} cells")

# Step 5: Save results
print("\n[5/5] Saving results...")
output_file = '/home/alan/Documents/decipher_batch_correction/simulation/archive/oldadata_alpha/adata_combined_2_delta_batch_corrected.h5ad'
adata.write(output_file)
print(f"✓ Results saved to: {output_file}")

# Print summary statistics
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

# Compute variance explained in each dimension
print("\nVariance explained by latent dimensions:")
z_var = np.var(results['z'], axis=0)
z_var_pct = 100 * z_var / z_var.sum()
for i, var_pct in enumerate(z_var_pct[:5]):  # Show top 5
    print(f"  Dimension {i+1}: {var_pct:.2f}%")

print("\nBatch attention strength statistics:")
print(f"  Mean: {adata.obs['batch_attention_strength'].mean():.4f}")
print(f"  Std:  {adata.obs['batch_attention_strength'].std():.4f}")
print(f"  Min:  {adata.obs['batch_attention_strength'].min():.4f}")
print(f"  Max:  {adata.obs['batch_attention_strength'].max():.4f}")

print("\nBatch attention by delta value:")
for delta_val in sorted(adata.obs['delta'].unique()):
    mask = adata.obs['delta'] == delta_val
    mean_attn = adata.obs.loc[mask, 'batch_attention_strength'].mean()
    print(f"  Delta={delta_val}: {mean_attn:.4f}")

print("\n" + "=" * 70)
print("OUTPUTS SAVED IN ANNDATA OBJECT")
print("=" * 70)
print("\n1. New embeddings in adata.obsm:")
print("   - 'X_decipher_batch_corrected_z' : Latent representation (10D)")
print("   - 'X_decipher_batch_corrected_v' : Component representation (2D)")

print("\n2. New observation in adata.obs:")
print("   - 'batch_attention_strength' : Attention weight per cell")
print("     (Higher = stronger batch effect for that cell)")

print("\n3. Model approach:")
print("   - Uses ATTENTION: z queries batch_emb via multi-head attention")
print("   - More complex than concatenation")
print("   - Slower training but potentially better mixing")
print("   - Provides interpretable attention weights")

print("\n" + "=" * 70)
print("NEXT STEPS")
print("=" * 70)
print("\n1. Compare with concatenation:")
print("   python compare_concat_vs_attention_delta.py")

print("\n2. Analyze attention patterns:")
print("   # High attention = strong batch effect on that cell")
print("   # Low attention = minimal batch effect")
print("   # Can identify cell types most affected by batch")

print("\n3. Decision:")
print("   - If attention wins by >0.05: use attention")
print("   - If similar performance: use concatenation (simpler)")
print("   - Check if added complexity is justified")

print("\n" + "=" * 70)
print("✓ ALL DONE!")
print("=" * 70)
