# Architecture Comparison: Concatenation vs. Attention

This document provides a visual comparison of both batch correction approaches.

## Side-by-Side Comparison

### Concatenation Approach (This Implementation)

```
┌─────────────────────────────────────────────────────────────┐
│                    CONCATENATION APPROACH                    │
└─────────────────────────────────────────────────────────────┘

Input: z (latent), batch_index

         ┌──────────┐
         │    z     │  (latent representation)
         │ [B, 10]  │
         └────┬─────┘
              │
              │         ┌──────────────┐
              │         │ batch_index  │
              │         │    [B]       │
              │         └──────┬───────┘
              │                │
              │                ▼
              │         ┌──────────────┐
              │         │  Embedding   │  Look up batch embedding
              │         │   Layer      │
              │         └──────┬───────┘
              │                │
              │                ▼
              │         ┌──────────────┐
              │         │  batch_emb   │
              │         │   [B, 64]    │
              │         └──────┬───────┘
              │                │
              └────────────────┘
                       │
                       │  CONCATENATE (simple!)
                       │
                       ▼
              ┌─────────────────┐
              │    combined     │
              │   [B, 10+64]    │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Linear(74,128) │
              │   + BatchNorm   │
              │   + ReLU        │
              │   + Dropout     │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Linear(128,256) │
              │   + BatchNorm   │
              │   + ReLU        │
              │   + Dropout     │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │Linear(256,genes)│
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   mu, theta     │  Gene expression parameters
              │  [B, genes]×2   │
              └─────────────────┘

Parameters: ~520K
Speed: ~8s/epoch
Complexity: Low ⭐
Flexibility: Medium
```

### Attention Approach (decipher-bc/)

```
┌─────────────────────────────────────────────────────────────┐
│                     ATTENTION APPROACH                       │
└─────────────────────────────────────────────────────────────┘

Input: z (latent), batch_index

         ┌──────────┐                  ┌──────────────┐
         │    z     │                  │ batch_index  │
         │ [B, 10]  │                  │    [B]       │
         └────┬─────┘                  └──────┬───────┘
              │                               │
              │                               ▼
              │                        ┌──────────────┐
              ▼                        │  Embedding   │
       ┌──────────────┐                │   Layer      │
       │ Projection   │                └──────┬───────┘
       │ z → 64 dim   │                       │
       └──────┬───────┘                       ▼
              │                        ┌──────────────┐
              ▼                        │  batch_emb   │
       ┌──────────────┐                │   [B, 64]    │
       │   z_proj     │                └──────┬───────┘
       │   [B, 64]    │                       │
       └──────┬───────┘                       │
              │                               │
              │    ┌──────────────────────────┘
              │    │
              │    │   MULTI-HEAD ATTENTION (complex!)
              │    │   z_proj queries batch_emb
              ▼    ▼
       ┌─────────────────────────┐
       │  MultiheadAttention     │
       │  • Query: z_proj        │
       │  • Key: batch_emb       │
       │  • Value: batch_emb     │
       │  • 4 attention heads    │
       └──────────┬──────────────┘
                  │
                  ▼
       ┌─────────────────┐
       │  attn_output    │  Batch info relevant to this cell
       │    [B, 64]      │
       └──────┬──────────┘
              │
              │         ┌──────────┐
              │         │    z     │
              │         │ [B, 10]  │
              │         └────┬─────┘
              │              │
              └──────────────┘
                       │
                       │  CONCATENATE
                       │
                       ▼
              ┌─────────────────┐
              │    combined     │
              │   [B, 10+64]    │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Linear(74,128) │
              │   + BatchNorm   │
              │   + ReLU        │
              │   + Dropout     │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Linear(128,256) │
              │   + BatchNorm   │
              │   + ReLU        │
              │   + Dropout     │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │Linear(256,genes)│
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   mu, theta     │  Gene expression parameters
              │  [B, genes]×2   │
              └─────────────────┘

Parameters: ~680K
Speed: ~12s/epoch
Complexity: High ⭐⭐⭐
Flexibility: High
```

## Key Differences

### 1. Batch Information Integration

**Concatenation**:
```python
# Fixed combination
batch_emb = embedding(batch_index)
combined = concat([z, batch_emb])  # Simple!
```

**Attention**:
```python
# Dynamic attention-based combination
batch_emb = embedding(batch_index)
z_proj = projection(z)

# z "asks": "What batch info is relevant for me?"
attn_output = attention(
    query=z_proj,      # "I'm looking for..."
    key=batch_emb,     # "Here's what's available..."
    value=batch_emb    # "Here's the actual information"
)
combined = concat([z, attn_output])  # Use attended info
```

### 2. What Makes Attention Special?

The attention mechanism allows each cell to **selectively focus** on relevant batch information:

```
Cell A (stem cell):
  "I'm a stem cell, batch effects mainly affect
   my proliferation genes"
  → Attention focuses on batch info relevant to proliferation

Cell B (differentiated):
  "I'm differentiated, batch effects mainly affect
   my housekeeping genes"
  → Attention focuses on different batch aspects

With concatenation, both cells get the SAME batch embedding.
With attention, each cell gets CUSTOMIZED batch information.
```

### 3. Information Flow

**Concatenation**:
```
z (cell state) ─┐
                 ├─→ concat ─→ MLP ─→ gene params
batch_emb ──────┘

Fixed relationship: batch info is the same for all cells
```

**Attention**:
```
z (cell state) ─→ projection ─→ query ─┐
                                        ├─→ attention ─→ customized batch info ─┐
batch_emb ──────────────────→ key/value ─┘                                       │
                                                                                  │
z (original) ────────────────────────────────────────────────────────────────────┤
                                                                                  │
                                                                                  ├─→ concat ─→ MLP ─→ gene params
                                                                                  │
                                                                                  ─┘

Dynamic relationship: each cell gets personalized batch correction
```

## Analogy: Restaurant Orders

### Concatenation Approach
```
Customer (cell) + Fixed Menu (batch info) → Chef (MLP) → Food (gene expression)

Every customer at table "Batch 1" gets the same menu.
The chef combines your preferences with the menu to make food.
```

### Attention Approach
```
Customer (cell) asks waiter (attention):
  "What items on the menu are good for someone like me?"

Waiter (attention) looks at:
  - Your preferences (query = z)
  - Menu items (key = batch_emb)
  - Actual dishes (value = batch_emb)

Waiter recommends personalized subset → Chef (MLP) → Food (gene expression)

Each customer gets a CUSTOMIZED recommendation even from the same menu.
```

## Parameter Breakdown

### Concatenation Model

```
Component                    Parameters
─────────────────────────────────────────
Batch Embedding (3×64)              192
MLP Layer 1 (74→128)             9,472
  ├─ Linear                      9,472
  └─ BatchNorm                     256
MLP Layer 2 (128→256)           32,768
  ├─ Linear                     32,768
  └─ BatchNorm                     512
Output Layer (256→genes×2)     ~512,000
─────────────────────────────────────────
TOTAL                          ~520,000
```

### Attention Model

```
Component                    Parameters
─────────────────────────────────────────
Batch Embedding (3×64)              192
Z Projection (10→64)                640
Multi-head Attention            ~49,000
  ├─ Query projection            4,096
  ├─ Key projection              4,096
  ├─ Value projection            4,096
  └─ Output projection           4,096
MLP Layer 1 (74→128)             9,472
  ├─ Linear                      9,472
  └─ BatchNorm                     256
MLP Layer 2 (128→256)           32,768
  ├─ Linear                     32,768
  └─ BatchNorm                     512
Output Layer (256→genes×2)     ~512,000
─────────────────────────────────────────
TOTAL                          ~680,000

DIFFERENCE: +160K parameters (mainly from attention)
```

## When Does Attention Help?

### Scenarios Where Attention Provides Value

1. **Complex Batch Effects**
   - Different cell types affected differently by batches
   - Multiple confounding factors
   - Non-linear batch interactions

2. **Interpretability Needed**
   - Want to see which cells are most affected by batch
   - Need to understand how correction works
   - Visualizing attention weights

3. **Heterogeneous Populations**
   - Multiple cell types with different batch sensitivities
   - Rare cell types that need special handling

### Scenarios Where Concatenation is Sufficient

1. **Simple Batch Effects**
   - Uniform technical variation
   - Single confounding factor
   - Linear batch effects

2. **Speed is Critical**
   - Development phase
   - Many hyperparameter experiments
   - Large-scale production

3. **Limited Resources**
   - Memory constraints
   - GPU limitations
   - Time constraints

## Mathematical Formulation

### Concatenation

```
Given:
  z ∈ ℝ^d_z          (latent)
  b ∈ {1,...,B}       (batch index)

1. Embed batch:
   e_b = Embedding(b) ∈ ℝ^d_e

2. Concatenate:
   h = [z; e_b] ∈ ℝ^(d_z + d_e)

3. Decode:
   μ, θ = MLP(h)

Parameters: O(d_z × d_e + MLP parameters)
```

### Attention

```
Given:
  z ∈ ℝ^d_z          (latent)
  b ∈ {1,...,B}       (batch index)

1. Embed batch:
   e_b = Embedding(b) ∈ ℝ^d_e

2. Project z to query space:
   q = W_q × z ∈ ℝ^d_e

3. Compute attention:
   α = softmax(q^T × e_b / √d_e)  (attention weights)
   a = α × e_b                     (attended batch info)

4. Concatenate:
   h = [z; a] ∈ ℝ^(d_z + d_e)

5. Decode:
   μ, θ = MLP(h)

Parameters: O(d_z × d_e + d_e² + MLP parameters)
Extra: d_e² for attention (significant when d_e is large)
```

## Visualization of Attention Weights

With the attention approach, you can visualize **which cells attend strongly to batch information**:

```python
# Get attention weights
attn_weights = model.get_attention_weights(z, batch_index)

# Visualize
import seaborn as sns
sns.heatmap(attn_weights, cmap='viridis')
plt.title('Attention to Batch Information')
plt.xlabel('Batch')
plt.ylabel('Cell')
```

This can reveal:
- Which cells are most batch-sensitive
- Which batches have strongest effects
- Cell-type-specific batch patterns

## Summary Table

| Aspect | Concatenation | Attention |
|--------|--------------|-----------|
| **Learnable integration** | No | Yes |
| **Cell-specific correction** | No | Yes |
| **Attention weights** | N/A | Available |
| **Extra parameters** | 0 | +160K |
| **Training time** | 1.0× | 1.5× |
| **Memory usage** | 1.0× | 1.25× |
| **Interpretability** | Simple | Rich |
| **Best for** | Baselines, speed | Performance, research |

## Recommendation

**Start with concatenation**, establish baseline performance, then **try attention** if:
- Baseline performance is insufficient
- You need interpretability
- You have complex batch structure
- You have computational resources

Use the `compare_approaches.py` script to make an informed decision!
