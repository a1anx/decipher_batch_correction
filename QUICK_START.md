# Batch-Corrected Decipher - Quick Start Guide

## 🚀 Quick Start (3 steps)

### Step 1: Configure
```python
from decipher_batch_corrected import DecipherBatchCorrectedConfig

config = DecipherBatchCorrectedConfig()
config.initialize_from_adata(adata, batch_key='batch')
```

### Step 2: Train
```python
from train_batch_corrected import train_batch_corrected_decipher

model, losses = train_batch_corrected_decipher(
    adata, batch_key='batch', config=config
)
```

### Step 3: Extract Embeddings
```python
from train_batch_corrected import evaluate_batch_correction

results = evaluate_batch_correction(model, adata, 'batch')
adata.obsm['X_decipher_z'] = results['z']
```

## 📋 Complete Example

```python
import scanpy as sc
from decipher_batch_corrected import DecipherBatchCorrectedConfig
from train_batch_corrected import train_batch_corrected_decipher, evaluate_batch_correction

# Load data
adata = sc.read_h5ad('your_data.h5ad')

# Configure (adjust these!)
config = DecipherBatchCorrectedConfig(
    dim_z=10,                       # Latent dimension
    dim_v=2,                        # Component dimension
    batch_emb_dim=64,               # Batch embedding size
    decoder_hidden_dims=[128, 256], # Decoder architecture
    n_attention_heads=4,            # Attention heads
    learning_rate=5e-3,
    batch_size=64,
    n_epochs=1000,
    early_stopping_patience=10
)
config.initialize_from_adata(adata, batch_key='batch')

# Train
model, losses = train_batch_corrected_decipher(
    adata, batch_key='batch', config=config, device='cuda'
)

# Evaluate and extract embeddings
results = evaluate_batch_correction(model, adata, batch_key='batch')
adata.obsm['X_decipher_z'] = results['z']
adata.obsm['X_decipher_v'] = results['v']

# Visualize
sc.pp.neighbors(adata, use_rep='X_decipher_z')
sc.tl.umap(adata)
sc.pl.umap(adata, color=['batch', 'cell_type'])
```

## 🔧 Key Parameters to Tune

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `batch_emb_dim` | 64 | 32-128 | Batch embedding capacity |
| `n_attention_heads` | 4 | 2-8 | Multi-head attention |
| `decoder_hidden_dims` | [128, 256] | [64,128] - [256,512] | Decoder capacity |
| `learning_rate` | 5e-3 | 1e-3 - 1e-2 | Training speed |
| `combination_mode` | "concat" | "concat"/"add" | How to combine z & batch |

## 📊 Visualize Attention

```python
# Attention shows batch effect strength per cell
adata.obs['batch_attention'] = results['attention_weights'].mean(axis=(1,2,3))
sc.pl.umap(adata, color='batch_attention')

# High attention = strong batch effect
# Low attention = weak batch effect
```

## 🔍 Architecture Overview

```
Input: [Gene Expression (x), Batch Index]
         ↓
    Encoder (unchanged)
         ↓
    Latent z, v
         ↓
    Decoder with Attention  ← NEW!
         ↓
    z queries batch embeddings
         ↓
    [z + batch_effect] → MLP → (mu, theta)
         ↓
Output: Gene Expression Parameters
```

## ✅ Checklist

- [ ] Installed: PyTorch, Pyro, NumPy
- [ ] Data has `adata.obs['batch']` column
- [ ] Configured model parameters
- [ ] Trained model
- [ ] Extracted embeddings
- [ ] Visualized results (UMAP, attention weights)

## 🐛 Common Issues

**"n_batches must be set"**
→ Call `config.initialize_from_adata(adata, batch_key='batch')`

**Out of Memory**
→ Reduce `batch_size`, `batch_emb_dim`, or `decoder_hidden_dims`

**Poor batch correction**
→ Increase `batch_emb_dim` or `n_attention_heads`

**Overfitting to batches**
→ Decrease `batch_emb_dim` or increase dropout

## 📁 Files Created

1. **batch_corrected_decoder.py** - Core attention decoder
2. **decipher_batch_corrected.py** - Full VAE model
3. **data_loader_batch_corrected.py** - Data utilities
4. **train_batch_corrected.py** - Training functions
5. **test_integration.py** - Integration tests
6. **INTEGRATION_SUMMARY.md** - Detailed documentation
7. **QUICK_START.md** - This file

## 🔗 Next Steps

1. Run on your data
2. Tune hyperparameters
3. Compare with original Decipher
4. Benchmark against other batch correction methods
5. Interpret attention weights

---

**Need help?** Check [INTEGRATION_SUMMARY.md](INTEGRATION_SUMMARY.md) for detailed documentation.
