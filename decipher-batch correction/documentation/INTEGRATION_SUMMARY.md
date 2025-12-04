# Batch-Corrected Decipher Integration Summary

## Overview

This document summarizes the integration of an attention-based batch correction mechanism into the Decipher VAE model, inspired by the Mr. VI architecture.

## Architecture Changes

### Original Decipher Architecture
```
Encoder (Guide):
  x → [encoder_x_to_z] → z
  [z, x] → [encoder_zx_to_v] → v

Decoder (Model):
  v → [decoder_v_to_z] → z
  z → [decoder_z_to_x] → x  ← Simple linear/MLP decoder
```

### New Batch-Corrected Architecture
```
Encoder (Guide): [UNCHANGED]
  x → [encoder_x_to_z] → z
  [z, x] → [encoder_zx_to_v] → v

Decoder (Model):
  v → [decoder_v_to_z] → z
  [z, batch_index] → [BatchCorrectedDecoder] → (mu, theta)  ← NEW!
                         ↓
                    [Attention Mechanism]
                         ↓
              Query: z (cell state)
              Key/Value: batch embeddings
```

## Files Created

### 1. `batch_corrected_decoder.py`
**Purpose**: Core attention-based decoder module

**Key Components**:
- `BatchCorrectedDecoder(nn.Module)`: Main decoder class
  - **Inputs**:
    - `z`: latent variable (batch_size, latent_dim)
    - `batch_index`: batch indices (batch_size,)
  - **Architecture**:
    - Batch embedding layer (learnable)
    - Multi-head attention (z queries batch embeddings)
    - Combination layer (concat or add)
    - MLP decoder (outputs mu and theta for Negative Binomial)
  - **Outputs**:
    - `mu`: mean gene expression (batch_size, n_genes)
    - `theta`: dispersion parameters (batch_size, n_genes)

**Key Methods**:
- `forward(z, batch_index)`: Main forward pass
- `get_attention_weights(z, batch_index)`: For interpretability

**Parameters**:
```python
BatchCorrectedDecoder(
    latent_dim=32,           # Dimension of z
    n_output=2000,           # Number of genes
    n_batches=5,             # Number of batches
    batch_emb_dim=64,        # Batch embedding dimension
    hidden_dims=[128, 256],  # MLP hidden layers
    n_heads=4,               # Attention heads
    dropout=0.1,
    combination_mode="concat"  # or "add"
)
```

### 2. `decipher_batch_corrected.py`
**Purpose**: Full VAE model with batch correction

**Key Components**:
- `DecipherBatchCorrectedConfig`: Configuration dataclass
  - Extends original config with batch correction parameters
  - `n_batches`: Number of unique batches (REQUIRED)
  - `batch_emb_dim`: Dimension of batch embeddings
  - `decoder_hidden_dims`: Hidden layers in decoder
  - `n_attention_heads`: Number of attention heads
  - `combination_mode`: How to combine z and attention output

- `DecipherBatchCorrected(nn.Module)`: Main model class
  - **Encoder**: Same as original Decipher
  - **Decoder**: Uses `BatchCorrectedDecoder` instead of `ConditionalDenseNN`

**Key Changes from Original**:
1. `decoder_z_to_x` is now `BatchCorrectedDecoder` (was `ConditionalDenseNN`)
2. `model()` method now accepts `batch_index` parameter
3. `guide()` method now accepts `batch_index` parameter
4. Negative Binomial parametrization updated (decoder outputs mu and theta directly)
5. New method: `get_batch_attention_weights()` for interpretability
6. Updated `impute_gene_expression_numpy()` to accept batch_index

**Configuration Example**:
```python
config = DecipherBatchCorrectedConfig(
    dim_z=10,                      # Latent dimension
    dim_v=2,                       # Component dimension
    n_batches=3,                   # Number of batches
    batch_emb_dim=64,              # Batch embedding dim
    decoder_hidden_dims=[128, 256], # Decoder MLP
    n_attention_heads=4,           # Attention heads
    learning_rate=5e-3,
    batch_size=64,
    n_epochs=1000
)
config.initialize_from_adata(adata, batch_key='batch')
```

### 3. `data_loader_batch_corrected.py`
**Purpose**: Data loading utilities for batch-corrected model

**Key Changes from Original**:
- Returns batch indices as **integers** (not one-hot encoded)
- Provides batch label mapping for reference

**Key Functions**:
- `make_batch_corrected_data_loader(adata, batch_key, ...)`
  - Returns: `(data_loader, batch_mapping)`
  - Data loader yields: `(genes, batch_indices)` tuples

- `make_train_val_loaders(adata, batch_key, val_frac=0.1, ...)`
  - Returns: `(train_loader, val_loader, batch_mapping)`
  - Properly splits data into train/val sets

**Usage Example**:
```python
train_loader, val_loader, batch_mapping = make_train_val_loaders(
    adata,
    batch_key='batch',
    batch_size=64,
    val_frac=0.1
)

# batch_mapping: {0: 'batch_A', 1: 'batch_B', 2: 'batch_C'}
```

### 4. `train_batch_corrected.py`
**Purpose**: Complete training script with Pyro SVI

**Key Functions**:
- `train_batch_corrected_decipher(adata, batch_key, config, device)`
  - Full training loop with validation
  - Early stopping support
  - Returns trained model and losses

- `evaluate_batch_correction(model, adata, batch_key, device)`
  - Computes embeddings (z, v)
  - Extracts attention weights
  - Returns evaluation metrics

**Usage Example**:
```python
model, losses = train_batch_corrected_decipher(
    adata,
    batch_key='batch',
    config=config,
    device='cuda'
)

# Evaluate
results = evaluate_batch_correction(model, adata, 'batch')
adata.obsm['X_decipher_z'] = results['z']
adata.obsm['X_decipher_v'] = results['v']
```

### 5. `test_integration.py`
**Purpose**: Integration test suite

**Tests**:
1. ✓ Module imports
2. ✓ Mock data creation
3. ✓ Configuration initialization
4. ✓ Model creation
5. ✓ Forward pass (encoder, decoder, attention)
6. ✓ Data loaders

## Key Differences from Original Decipher

### 1. **Decoder Architecture**
| Original | Batch-Corrected |
|----------|-----------------|
| `ConditionalDenseNN(z -> x)` | `BatchCorrectedDecoder([z, batch] -> x)` |
| Simple MLP | Attention + MLP |
| No batch modeling | Explicit batch correction |

### 2. **Output Parametrization**
| Original | Batch-Corrected |
|----------|-----------------|
| Outputs: logits | Outputs: (mu, theta) |
| Uses softmax normalization | Direct NB parameters |
| Single theta per gene | Batch-specific theta possible |

### 3. **Training Data**
| Original | Batch-Corrected |
|----------|-----------------|
| Data: (genes, [context]) | Data: (genes, batch_indices) |
| Context: one-hot encoded | Batch: integer indices |
| Optional context | Required batch_index |

### 4. **Model Signature**
```python
# Original
model.model(x, context=None)
model.guide(x, context=None)

# Batch-Corrected
model.model(x, batch_index, context=None)
model.guide(x, batch_index, context=None)
```

## Usage Instructions

### Step 1: Prepare Your Data
```python
import scanpy as sc

# Load your data
adata = sc.read_h5ad('your_data.h5ad')

# Ensure you have a batch column in adata.obs
# Example: adata.obs['batch'] = ['batch_1', 'batch_2', ...]
```

### Step 2: Configure the Model
```python
from decipher_batch_corrected import DecipherBatchCorrectedConfig

config = DecipherBatchCorrectedConfig(
    dim_z=10,                       # Latent dimension
    dim_v=2,                        # Component dimension
    batch_emb_dim=64,               # Batch embedding dimension
    decoder_hidden_dims=[128, 256], # Decoder architecture
    n_attention_heads=4,            # Number of attention heads
    combination_mode="concat",      # Combination mode
    learning_rate=5e-3,
    batch_size=64,
    n_epochs=1000,
    early_stopping_patience=10
)

# Initialize from your data
config.initialize_from_adata(adata, batch_key='batch')
```

### Step 3: Train the Model
```python
from train_batch_corrected import train_batch_corrected_decipher

model, losses = train_batch_corrected_decipher(
    adata,
    batch_key='batch',
    config=config,
    device='cuda'  # or 'cpu'
)
```

### Step 4: Extract Embeddings
```python
from train_batch_corrected import evaluate_batch_correction

results = evaluate_batch_correction(model, adata, batch_key='batch')

# Add to AnnData
adata.obsm['X_decipher_z'] = results['z']
adata.obsm['X_decipher_v'] = results['v']
adata.obs['batch_attention'] = results['attention_weights'].mean(axis=(1,2,3))
```

### Step 5: Visualize Results
```python
import scanpy as sc

# UMAP on batch-corrected embeddings
sc.pp.neighbors(adata, use_rep='X_decipher_z')
sc.tl.umap(adata)
sc.pl.umap(adata, color=['batch', 'cell_type', 'batch_attention'])
```

## Hyperparameter Tuning Guide

### Critical Parameters

1. **`batch_emb_dim`** (default: 64)
   - Controls capacity of batch embedding
   - Larger = more expressive batch modeling
   - Smaller = less risk of overfitting
   - Recommended: 32-128

2. **`n_attention_heads`** (default: 4)
   - Multi-head attention heads
   - More heads = different aspects of batch effects
   - Recommended: 2-8

3. **`decoder_hidden_dims`** (default: [128, 256])
   - Architecture of final MLP
   - Deeper/wider = more capacity
   - Recommended: [64, 128] to [256, 512]

4. **`combination_mode`** (default: "concat")
   - "concat": Preserves both z and batch info
   - "add": More parameter efficient, requires matching dims
   - Recommended: "concat" for most cases

5. **`learning_rate`** (default: 5e-3)
   - Higher = faster but less stable
   - Lower = slower but more stable
   - Recommended: 1e-3 to 1e-2

## Interpretation: Attention Weights

The attention weights reveal how strongly each cell's latent state relies on batch-specific information:

```python
# Get attention weights
attn_weights = model.get_batch_attention_weights(x, batch_index)
# Shape: (batch_size, n_heads, 1, 1)

# Average attention strength per cell
avg_attention = attn_weights.mean(axis=(1, 2, 3))

# High attention = strong batch effect
# Low attention = weak batch effect
```

**Visualization**:
```python
adata.obs['batch_correction_strength'] = avg_attention
sc.pl.umap(adata, color='batch_correction_strength')
```

## Advantages Over Original Decipher

1. **Explicit Batch Modeling**: Directly models batch effects rather than ignoring them
2. **Interpretability**: Attention weights show which cells are affected by batch
3. **Flexibility**: Can handle varying number of batches without architectural changes
4. **Disentanglement**: Separates biological signal (z) from technical variation (batch)
5. **Scalability**: Attention mechanism scales better than concatenation for many batches

## Comparison with Mr. VI

This implementation is **inspired by** Mr. VI but adapted for Decipher:

| Feature | Mr. VI | This Implementation |
|---------|--------|---------------------|
| Batch Correction | Attention-based | ✓ Attention-based |
| Architecture | Style/Content VAE | Hierarchical VAE (v→z→x) |
| Encoder | Modified | ✓ Unchanged from Decipher |
| Decoder | Attention in decoder | ✓ Attention in z→x |
| Framework | PyTorch | ✓ PyTorch + Pyro |

## Testing

Run the integration test:
```bash
python test_integration.py
```

Requirements:
- PyTorch
- Pyro-ppl
- NumPy
- (Optional) scanpy for real data

## Troubleshooting

### Issue: "n_batches must be set"
**Solution**: Call `config.initialize_from_adata(adata, batch_key='batch')` before creating model

### Issue: Attention dimension mismatch
**Solution**: Ensure `latent_dim` is compatible. Use `combination_mode="concat"` (default) if unsure

### Issue: Out of memory
**Solutions**:
- Reduce `batch_size`
- Reduce `batch_emb_dim`
- Reduce `decoder_hidden_dims`
- Use gradient checkpointing (not implemented yet)

### Issue: Poor batch correction
**Solutions**:
- Increase `batch_emb_dim`
- Increase `n_attention_heads`
- Check that batch labels are correct in `adata.obs[batch_key]`
- Visualize attention weights to diagnose

### Issue: Overfitting to batches
**Solutions**:
- Decrease `batch_emb_dim`
- Increase dropout
- Use regularization on embeddings
- Reduce model capacity

## Next Steps

1. **Test on your data**: Use your actual single-cell dataset
2. **Tune hyperparameters**: Adjust based on your data characteristics
3. **Evaluate batch correction**: Compare UMAPs before/after
4. **Benchmark**: Compare against other batch correction methods (Harmony, scVI, etc.)
5. **Extend**: Add additional covariates (cell type, donor, etc.)

## File Structure

```
decipher_batch_correction/
├── batch_corrected_decoder.py          # Core decoder module
├── decipher_batch_corrected.py         # Full VAE model
├── data_loader_batch_corrected.py      # Data utilities
├── train_batch_corrected.py            # Training script
├── test_integration.py                 # Integration tests
├── INTEGRATION_SUMMARY.md              # This file
└── decipher-main/                      # Original Decipher code
    └── decipher/
        └── tools/
            └── _decipher/
                ├── decipher.py         # Original model
                ├── module.py           # ConditionalDenseNN
                └── data.py             # Original data loader
```

## Citation

If you use this batch-corrected implementation, please cite:
- Original Decipher paper
- Mr. VI paper (for batch correction inspiration)
- This implementation (if published)

## Contact & Support

For questions or issues:
1. Check this documentation
2. Run `test_integration.py` to verify setup
3. Check attention weights for debugging
4. Adjust hyperparameters systematically

---

**Summary**: This integration adds attention-based batch correction to Decipher's decoder while preserving the hierarchical VAE structure. The attention mechanism allows the model to learn which cells are affected by batch effects and to what degree, providing both better batch correction and interpretability.
