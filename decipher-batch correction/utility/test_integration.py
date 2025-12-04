"""
Integration test for batch-corrected Decipher model.

This script tests that all components work together correctly.
"""

import numpy as np
import pandas as pd
import torch
import sys

print("=" * 60)
print("BATCH-CORRECTED DECIPHER INTEGRATION TEST")
print("=" * 60)

# Test 1: Import all modules
print("\n[1/6] Testing imports...")
try:
    from batch_corrected_decoder import BatchCorrectedDecoder
    from decipher_batch_corrected import DecipherBatchCorrected, DecipherBatchCorrectedConfig
    from data_loader_batch_corrected import make_batch_corrected_data_loader, make_train_val_loaders
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Create mock data
print("\n[2/6] Creating mock AnnData...")
try:
    class MockAnnData:
        def __init__(self, n_cells, n_genes, n_batches):
            self.shape = (n_cells, n_genes)
            self.X = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes)).astype(np.float32)
            self.obs = pd.DataFrame({
                'batch': [f'batch_{i % n_batches}' for i in range(n_cells)]
            })

    n_cells = 200
    n_genes = 100
    n_batches = 3
    mock_adata = MockAnnData(n_cells, n_genes, n_batches)
    print(f"✓ Created mock data: {n_cells} cells, {n_genes} genes, {n_batches} batches")
except Exception as e:
    print(f"✗ Mock data creation failed: {e}")
    sys.exit(1)

# Test 3: Initialize configuration
print("\n[3/6] Initializing configuration...")
try:
    config = DecipherBatchCorrectedConfig(
        dim_z=8,
        dim_v=2,
        batch_emb_dim=16,
        decoder_hidden_dims=[32, 64],
        n_attention_heads=2,
        batch_size=32,
        n_epochs=5
    )
    config.initialize_from_adata(mock_adata, batch_key='batch')
    print(f"✓ Config initialized:")
    print(f"  - dim_z: {config.dim_z}")
    print(f"  - dim_genes: {config.dim_genes}")
    print(f"  - n_batches: {config.n_batches}")
except Exception as e:
    print(f"✗ Configuration failed: {e}")
    sys.exit(1)

# Test 4: Create model
print("\n[4/6] Creating model...")
try:
    model = DecipherBatchCorrected(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {n_params:,} parameters")
    print(f"  - Encoder x->z: {sum(p.numel() for p in model.encoder_x_to_z.parameters()):,} params")
    print(f"  - Encoder zx->v: {sum(p.numel() for p in model.encoder_zx_to_v.parameters()):,} params")
    print(f"  - Decoder v->z: {sum(p.numel() for p in model.decoder_v_to_z.parameters()):,} params")
    print(f"  - Decoder z->x (batch-corrected): {sum(p.numel() for p in model.decoder_z_to_x.parameters()):,} params")
except Exception as e:
    print(f"✗ Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test forward pass
print("\n[5/6] Testing forward pass...")
try:
    # Create test batch
    batch_size = 16
    x_test = torch.FloatTensor(mock_adata.X[:batch_size])
    batch_indices_test = torch.LongTensor([i % n_batches for i in range(batch_size)])

    # Test encoder (guide)
    z_loc, v_loc, z_scale, v_scale = model.guide(x_test, batch_indices_test)
    print(f"✓ Encoder (guide) works:")
    print(f"  - z_loc shape: {z_loc.shape}")
    print(f"  - v_loc shape: {v_loc.shape}")

    # Test decoder
    mu, theta = model.decoder_z_to_x(z_loc, batch_indices_test)
    print(f"✓ Decoder (z->x with batch correction) works:")
    print(f"  - mu shape: {mu.shape}")
    print(f"  - theta shape: {theta.shape}")

    # Test attention weights
    attn_weights = model.get_batch_attention_weights(x_test, batch_indices_test)
    print(f"✓ Attention mechanism works:")
    print(f"  - attention weights shape: {attn_weights.shape}")

    # Test imputation
    imputed = model.impute_gene_expression_numpy(x_test.numpy(), batch_indices_test.numpy())
    print(f"✓ Imputation works:")
    print(f"  - imputed shape: {imputed.shape}")

except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test data loaders
print("\n[6/6] Testing data loaders...")
try:
    # Test single data loader
    data_loader, batch_mapping = make_batch_corrected_data_loader(
        mock_adata,
        batch_key='batch',
        batch_size=32
    )
    genes_batch, batch_indices = next(iter(data_loader))
    print(f"✓ Single data loader works:")
    print(f"  - genes batch shape: {genes_batch.shape}")
    print(f"  - batch indices shape: {batch_indices.shape}")
    print(f"  - batch mapping: {batch_mapping}")

    # Test train/val split
    train_loader, val_loader, _ = make_train_val_loaders(
        mock_adata,
        batch_key='batch',
        batch_size=32,
        val_frac=0.2
    )
    print(f"✓ Train/val split works:")
    print(f"  - train batches: {len(train_loader)}")
    print(f"  - val batches: {len(val_loader)}")

except Exception as e:
    print(f"✗ Data loader test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("ALL TESTS PASSED! ✓")
print("=" * 60)
print("\nThe batch-corrected Decipher model is ready to use!")
print("\nKey files created:")
print("  1. batch_corrected_decoder.py - Attention-based decoder")
print("  2. decipher_batch_corrected.py - Full VAE model")
print("  3. data_loader_batch_corrected.py - Data loading utilities")
print("  4. train_batch_corrected.py - Training script")
print("\nNext steps:")
print("  - Run train_batch_corrected.py with your data")
print("  - Adjust hyperparameters in DecipherBatchCorrectedConfig")
print("  - Visualize attention weights to interpret batch effects")
