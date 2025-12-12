"""
Run concatenation-based batch-corrected Decipher on adata_combined_2_delta.h5ad

This script:
1. Loads the delta shift data
2. Uses 'delta' column as batch information
3. Trains the concatenation-based batch-corrected Decipher model
4. Extracts embeddings
5. Saves results back to the AnnData object

This is the SIMPLER baseline approach compared to attention.
"""

import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from decipher_batch_corrected_concat import DecipherBatchCorrectedConcatConfig
from train_batch_corrected_concat import train_batch_corrected_decipher_concat, evaluate_batch_correction_concat

print("=" * 70)
print("CONCATENATION-BASED BATCH CORRECTION - Training on Delta Shift Data")
print("=" * 70)
print("This is the SIMPLER baseline approach (no attention mechanism)")
print("=" * 70)

# Step 1: Load data
print("\n[1/5] Loading data...")
adata = sc.read_h5ad('/home/alan/Documents/decipher_batch_correction/simulation/archive/oldadata_alpha/adata_combined_2_delta.h5ad')
print(f"✓ Loaded data: {adata.shape[0]} cells, {adata.shape[1]} genes")
print(f"  Batch column: 'delta' with {len(adata.obs['delta'].unique())} conditions")
print(f"  Conditions: {sorted(list(adata.obs['delta'].unique()))}")

# Step 2: Configure model
print("\n[2/5] Configuring model...")
config = DecipherBatchCorrectedConcatConfig(
    # Latent dimensions
    dim_z=10,                       # Match original Decipher
    dim_v=2,                        # Match original Decipher

    # Batch correction parameters
    n_batches=None,                 # Will be set from data
    batch_emb_dim=32,               # Batch embedding dimension (same as attention)
    decoder_hidden_dims=[64, 128],  # Decoder MLP architecture (same as attention)

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
print(f"  - Approach: CONCATENATION (simpler than attention)")

# Step 3: Train model
print("\n[3/5] Training model...")
print(f"  Training for up to {config.n_epochs} epochs (early stopping: {config.early_stopping_patience})")
print(f"  This may take several minutes...")

model, losses = train_batch_corrected_decipher_concat(
    adata,
    batch_key='delta',
    config=config,
    device='cpu'  # Use CPU
)

print(f"✓ Training complete!")
print(f"  Final training loss: {losses['train_losses'][-1]:.2f}")
print(f"  Final validation loss: {losses['val_losses'][-1]:.2f}")
print(f"  Trained for {len(losses['train_losses'])} epochs")

# Step 4: Extract embeddings
print("\n[4/5] Extracting embeddings...")
results = evaluate_batch_correction_concat(model, adata, batch_key='delta', device='cpu')

# Add to AnnData
adata.obsm['X_decipher_concat_z'] = results['z']
adata.obsm['X_decipher_concat_v'] = results['v']

print(f"✓ Embeddings extracted:")
print(f"  - X_decipher_concat_z: {results['z'].shape}")
print(f"  - X_decipher_concat_v: {results['v'].shape}")

# Step 5: Save results
print("\n[5/5] Saving results...")
output_file = '/home/alan/Documents/decipher_batch_correction/simulation/archive/oldadata_alpha/adata_combined_2_delta_concat.h5ad'
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

print("\n" + "=" * 70)
print("OUTPUTS SAVED IN ANNDATA OBJECT")
print("=" * 70)
print("\n1. New embeddings in adata.obsm:")
print("   - 'X_decipher_concat_z' : Latent representation (10D)")
print("   - 'X_decipher_concat_v' : Component representation (2D)")

print("\n2. Model approach:")
print("   - Uses CONCATENATION: concat([z, batch_emb]) → MLP")
print("   - Simpler than attention (no multi-head attention)")
print("   - Faster training")
print("   - Good baseline for comparison")

print("\n" + "=" * 70)
print("NEXT STEPS")
print("=" * 70)
print("\n1. Visualize embeddings:")
print("   python visualize_all_concat_results_delta.py")

print("\n2. Compare with attention approach (if available):")
print("   python compare_concat_vs_attention_delta.py")

print("\n3. Analyze performance:")
print("   - Check if batch mixing is good enough")
print("   - Verify biological signals are preserved")
print("   - Decide if attention is needed for better results")

print("\n" + "=" * 70)
print("✓ ALL DONE!")
print("=" * 70)
