# Guide (Inference) and Model (Generative) Path Modifications

## ✅ Answer: YES, Both Guide and Model Are Modified

Both the **guide** (inference/encoder path) and **model** (generative/decoder path) now accept `batch_index` as input.

However, the modifications are **asymmetric**:
- **Model (generative)**: Batch correction applied in decoder (z→x)
- **Guide (inference)**: Batch index passed through but encoder architecture unchanged

---

## Detailed Comparison

### Original Decipher

```python
class Decipher(nn.Module):
    def model(self, x, context=None):
        # Generative path: p(x, z, v)
        v ~ p(v)                    # Prior on v
        z ~ p(z|v)                  # v → z decoder
        x ~ p(x|z)                  # z → x decoder (simple MLP)

    def guide(self, x, context=None):
        # Inference path: q(z, v | x)
        z ~ q(z|x)                  # x → z encoder
        v ~ q(v|z,x)                # [z,x] → v encoder
```

### Batch-Corrected Decipher

```python
class DecipherBatchCorrected(nn.Module):
    def model(self, x, batch_index, context=None):
        # Generative path: p(x, z, v | batch)
        v ~ p(v)                    # Prior on v (UNCHANGED)
        z ~ p(z|v)                  # v → z decoder (UNCHANGED)
        x ~ p(x|z, batch)          # z → x decoder (CHANGED ← attention!)

    def guide(self, x, batch_index, context=None):
        # Inference path: q(z, v | x, batch)
        z ~ q(z|x)                  # x → z encoder (UNCHANGED)
        v ~ q(v|z,x)                # [z,x] → v encoder (UNCHANGED)
```

---

## Modification Details

### 1. Model (Generative Path) - MODIFIED ✓

#### Original Code
```python
def model(self, x, context=None):
    pyro.module("decipher", self)

    # ... prior on v ...
    v = pyro.sample("v", prior)

    # v → z
    z_loc, z_scale = self.decoder_v_to_z(v, context=context)
    z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

    # z → x (SIMPLE MLP)
    mu = self.decoder_z_to_x(z, context=context)
    mu = softmax(mu, dim=-1)
    library_size = x.sum(axis=-1, keepdim=True)

    logit = torch.log(library_size * mu + ε) - torch.log(theta + ε)
    x_dist = dist.NegativeBinomial(total_count=theta + ε, logits=logit)
    pyro.sample("x", x_dist.to_event(1), obs=x)
```

#### Batch-Corrected Code
```python
def model(self, x, batch_index, context=None):  # ← batch_index added
    pyro.module("decipher", self)

    # ... prior on v (UNCHANGED) ...
    v = pyro.sample("v", prior)

    # v → z (UNCHANGED)
    z_loc, z_scale = self.decoder_v_to_z(v, context=context)
    z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

    # z → x (BATCH-CORRECTED WITH ATTENTION)
    mu, theta = self.decoder_z_to_x(z, batch_index)  # ← NEW!
    #           ↑ BatchCorrectedDecoder with attention

    # Negative Binomial with directly output parameters
    logit = torch.log(mu + ε) - torch.log(theta + ε)
    x_dist = dist.NegativeBinomial(total_count=theta + ε, logits=logit)
    pyro.sample("x", x_dist.to_event(1), obs=x)
```

**Key Changes:**
1. ✓ Accepts `batch_index` parameter
2. ✓ Decoder `z→x` replaced with `BatchCorrectedDecoder`
3. ✓ Attention mechanism: z queries batch embeddings
4. ✓ Outputs (mu, theta) directly (no softmax/library size)

---

### 2. Guide (Inference Path) - SIGNATURE MODIFIED, ARCHITECTURE UNCHANGED

#### Original Code
```python
def guide(self, x, context=None):
    pyro.module("decipher", self)

    with pyro.plate("batch", len(x)):
        x = torch.log1p(x)

        # x → z (ENCODER)
        z_loc, z_scale = self.encoder_x_to_z(x, context=context)
        z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

        # [z, x] → v (ENCODER)
        zx = torch.cat([z, x], dim=-1)
        v_loc, v_scale = self.encoder_zx_to_v(zx, context=context)
        pyro.sample("v", dist.Normal(v_loc, v_scale).to_event(1))

    return z_loc, v_loc, z_scale, v_scale
```

#### Batch-Corrected Code
```python
def guide(self, x, batch_index, context=None):  # ← batch_index added
    pyro.module("decipher", self)

    with pyro.plate("batch", len(x)):
        x_transformed = torch.log1p(x)

        # x → z (ENCODER - UNCHANGED)
        z_loc, z_scale = self.encoder_x_to_z(x_transformed, context=context)
        z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

        # [z, x] → v (ENCODER - UNCHANGED)
        zx = torch.cat([z, x_transformed], dim=-1)
        v_loc, v_scale = self.encoder_zx_to_v(zx, context=context)
        pyro.sample("v", dist.Normal(v_loc, v_scale).to_event(1))

    return z_loc, v_loc, z_scale, v_scale
```

**Key Changes:**
1. ✓ Accepts `batch_index` parameter (for consistency)
2. ✗ `batch_index` NOT used in encoder computations
3. ✗ Encoder architecture completely unchanged
4. ✗ No attention in inference path

**Why is batch_index not used in guide?**
- The encoder learns to map x → z directly
- Batch correction happens in the generative path (z → x)
- The inference network doesn't need explicit batch modeling
- z learned by encoder should already be batch-invariant

---

## Path-by-Path Breakdown

### Generative Path (Model): p(x, z, v | batch)

```
                    UNCHANGED    UNCHANGED        CHANGED
                        ↓            ↓               ↓
Prior p(v) → Sample v → Decode v→z → Sample z → Decode z→x → Observe x
                                                      ↓
                                           BatchCorrectedDecoder
                                                      ↓
                                           ┌─────────────────┐
                                           │ z (query)       │
                                           │      ×          │
                                           │ batch (key/val) │
                                           │      ↓          │
                                           │  Attention      │
                                           │      ↓          │
                                           │ [z, batch_eff]  │
                                           │      ↓          │
                                           │     MLP         │
                                           │      ↓          │
                                           │  (mu, theta)    │
                                           └─────────────────┘
```

**Batch correction applied here!** ✓

### Inference Path (Guide): q(z, v | x, batch)

```
                UNCHANGED      UNCHANGED
                    ↓              ↓
Observe x → Encode x→z → Sample z → Encode [z,x]→v → Sample v
               ↓                        ↓
      ConditionalDenseNN       ConditionalDenseNN
         (from original)          (from original)
```

**No batch correction in encoders** (batch_index parameter ignored)

---

## Why This Design?

### Asymmetric Batch Correction

This design follows the principle that:
1. **Batch effects are observational artifacts** in the generative process
2. **The encoder should learn batch-invariant representations**
3. **The decoder accounts for batch-specific technical variation**

### Analogy: Image Generation with Style Transfer

```
Content (z) = What the image shows (biological signal)
Style (batch) = How it's rendered (technical variation)

Encoder:  Image → Content (ignores style)
Decoder:  Content + Style → Image (applies style)
```

In our case:
```
Biological Signal (z) = Cell state
Technical Variation (batch) = Batch effects

Encoder:  Expression → Cell State (batch-invariant)
Decoder:  Cell State + Batch → Expression (batch-specific)
```

---

## Comparison Table

| Component | Original | Batch-Corrected | Batch Used? |
|-----------|----------|-----------------|-------------|
| **Model** | | | |
| v prior | p(v) | p(v) | No |
| v→z decoder | ConditionalDenseNN | ConditionalDenseNN | No |
| z→x decoder | ConditionalDenseNN | **BatchCorrectedDecoder** | **Yes** ✓ |
| **Guide** | | | |
| x→z encoder | ConditionalDenseNN | ConditionalDenseNN | No |
| [z,x]→v encoder | ConditionalDenseNN | ConditionalDenseNN | No |

---

## Signature Comparison

### Function Signatures

```python
# Original
def model(self, x, context=None) → None
def guide(self, x, context=None) → (z_loc, v_loc, z_scale, v_scale)

# Batch-Corrected
def model(self, x, batch_index, context=None) → None
def guide(self, x, batch_index, context=None) → (z_loc, v_loc, z_scale, v_scale)
```

### Training Loop

```python
# Original
for genes, context in data_loader:
    loss = svi.step(genes, context)

# Batch-Corrected
for genes, batch_indices in data_loader:
    loss = svi.step(genes, batch_indices)
```

---

## Summary

| Question | Answer |
|----------|--------|
| Does it modify the model (generative)? | **YES** ✓ - Batch correction in z→x decoder |
| Does it modify the guide (inference)? | **Partially** - Signature changed, architecture unchanged |
| Are both paths batch-aware? | **Model: Yes**, **Guide: No** |
| Where is attention applied? | **Only in decoder** (generative path) |
| Are encoders changed? | **NO** - Reuses original encoders |

---

## Practical Implications

### During Training
- **Forward pass (guide)**: Encodes x → z, batch_index passed but not used
- **Generative pass (model)**: Decodes z + batch_index → x with attention

### During Inference
```python
# Get latent representation (batch-invariant)
z_loc, v_loc, _, _ = model.guide(x, batch_index)

# Generate with specific batch
mu, theta = model.decoder_z_to_x(z, batch_index)

# Can generate same z with different batches!
mu_batch1 = model.decoder_z_to_x(z, torch.tensor([0]))
mu_batch2 = model.decoder_z_to_x(z, torch.tensor([1]))
# mu_batch1 ≠ mu_batch2 (different batch effects!)
```

This allows you to:
1. **Remove batch effects**: Use z (batch-invariant latent)
2. **Simulate batch transfer**: Generate same cell in different batches
3. **Interpret batch effects**: Compare outputs for same z across batches

---

**In conclusion**: Both `model` and `guide` accept `batch_index`, but only the `model` (generative path) actively uses it for batch correction via the attention mechanism in the decoder.
