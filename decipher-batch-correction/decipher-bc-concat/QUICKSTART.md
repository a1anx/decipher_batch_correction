# Quick Start: Concatenation-Based Batch Correction

This is the **simple baseline** approach for batch correction in Decipher.

## TL;DR

```bash
# Test the implementation
python batch_corrected_decoder_concat.py
python decipher_batch_corrected_concat.py

# Test on your data (adata_combined_2.h5ad)
python run_concat_on_data_2.py
python visualize_all_concat_results.py
python compare_concat_vs_attention.py
```

## Why This Approach?

Your TA suggested trying concatenation first because:

1. **Simpler** - Easier to understand and debug
2. **Faster** - Trains ~1.5× faster than attention
3. **Baseline** - Establishes performance floor
4. **Good enough?** - Often sufficient for simple batch effects

## What's Different from Attention?

```python
# CONCATENATION (this approach)
combined = concat([z, batch_embedding])
output = mlp(combined)

# ATTENTION (decipher-bc/)
attn_output = attention(query=z, key=batch_embedding, value=batch_embedding)
combined = concat([z, attn_output])
output = mlp(combined)
```

**Concatenation**: Every cell gets the same batch embedding
**Attention**: Each cell gets customized batch information

## Minimal Example

```python
import scanpy as sc
from decipher_batch_corrected_concat import DecipherBatchCorrectedConcatConfig
from train_batch_corrected_concat import train_batch_corrected_decipher_concat

# Load data
adata = sc.read_h5ad("your_data.h5ad")

# Train with defaults
model, losses = train_batch_corrected_decipher_concat(
    adata,
    batch_key='batch'
)

# Get embeddings
from train_batch_corrected_concat import evaluate_batch_correction_concat
results = evaluate_batch_correction_concat(model, adata, 'batch')

# Add to adata
adata.obsm['X_decipher_z'] = results['z']
```

## When to Switch to Attention?

Try attention (`../decipher-bc/`) if:
- Concatenation baseline isn't good enough
- You need interpretability (attention weights)
- You have complex batch effects
- You have time for longer training

## Compare Both

```python
# This will train both and compare them
python compare_approaches.py
```

Creates comparison plots and tells you which to use.

## Files

- `batch_corrected_decoder_concat.py` - The decoder architecture
- `decipher_batch_corrected_concat.py` - Full model
- `train_batch_corrected_concat.py` - Training utilities
- `run_concat_on_data_2.py` - Train on your data
- `visualize_all_concat_results.py` - All visualizations
- `compare_concat_vs_attention.py` - Compare with attention
- `README.md` - Full documentation
- `ARCHITECTURE_COMPARISON.md` - Visual comparison

## Next Steps

1. ✅ Test on your data
2. ✅ Compare with attention approach
3. ✅ Pick the best approach for your use case
