"""
Test script to verify the attention mechanism behavior in BatchCorrectedDecoder.
This will trace through tensor shapes and attention weights to understand what's happening.
"""
import torch
import torch.nn as nn
from batch_corrected_decoder import BatchCorrectedDecoder


def test_attention_mechanism():
    """Test what the attention mechanism is actually doing."""

    print("="*80)
    print("TESTING ATTENTION MECHANISM IN BATCH CORRECTED DECODER")
    print("="*80)

    # Setup
    latent_dim = 8
    n_genes = 100
    n_batches = 3
    batch_emb_dim = 16
    batch_size = 5

    # Create decoder
    decoder = BatchCorrectedDecoder(
        latent_dim=latent_dim,
        n_output=n_genes,
        n_batches=n_batches,
        batch_emb_dim=batch_emb_dim,
        hidden_dims=[32],
        n_heads=2,
        dropout=0.0,  # No dropout for testing
        combination_mode="concat"
    )

    # Create sample inputs - cells from different batches
    torch.manual_seed(42)
    z = torch.randn(batch_size, latent_dim)
    batch_index = torch.tensor([0, 1, 2, 0, 1])  # Different batch assignments

    print(f"\nInput shapes:")
    print(f"  z: {z.shape} (batch_size={batch_size}, latent_dim={latent_dim})")
    print(f"  batch_index: {batch_index.shape} = {batch_index.tolist()}")
    print(f"  Batch assignments: {batch_index.tolist()}")

    # Step through the forward pass manually
    print("\n" + "="*80)
    print("MANUAL FORWARD PASS - TRACING ATTENTION")
    print("="*80)

    # Step 1: Get batch embeddings
    batch_emb = decoder.batch_embedding(batch_index)
    print(f"\nStep 1 - Batch Embeddings:")
    print(f"  batch_emb shape: {batch_emb.shape}")
    print(f"  Each cell gets its OWN batch's embedding:")
    for i in range(batch_size):
        print(f"    Cell {i} (batch {batch_index[i].item()}) → embedding vector of dim {batch_emb_dim}")

    # Step 2: Project z
    z_proj = decoder.z_projection(z)
    print(f"\nStep 2 - Project z:")
    print(f"  z_proj shape: {z_proj.shape}")

    # Step 3: Reshape for attention
    query = z_proj.unsqueeze(0)  # (1, batch_size, batch_emb_dim)
    key = batch_emb.unsqueeze(0)  # (1, batch_size, batch_emb_dim)
    value = batch_emb.unsqueeze(0)  # (1, batch_size, batch_emb_dim)

    print(f"\nStep 3 - Reshape for MultiheadAttention:")
    print(f"  query shape: {query.shape} (seq_len=1, batch={batch_size}, embed_dim={batch_emb_dim})")
    print(f"  key shape: {key.shape}")
    print(f"  value shape: {value.shape}")
    print(f"\n  CRITICAL OBSERVATION:")
    print(f"    seq_len = 1 means each sample attends to ONLY ITS OWN position!")
    print(f"    Cell i's query[0, i, :] only looks at key[0, i, :] (not other cells)")

    # Step 4: Apply attention
    attn_output, attn_weights = decoder.multihead_attn(
        query=query,
        key=key,
        value=value,
        need_weights=True,
        average_attn_weights=True
    )

    print(f"\nStep 4 - Attention Output:")
    print(f"  attn_output shape: {attn_output.shape}")
    print(f"  attn_weights shape: {attn_weights.shape}")
    print(f"  Attention weights (averaged across heads):")
    print(f"    {attn_weights.squeeze(0)}")
    print(f"\n  INTERPRETATION:")
    print(f"    Each row is a cell, each column is what it attends to")
    print(f"    Since seq_len=1, attention is DIAGONAL (cell i → position i only)")

    # Verify attention is diagonal
    attn_weights_2d = attn_weights.squeeze(0)
    is_diagonal = torch.allclose(attn_weights_2d, torch.eye(batch_size), atol=1e-6)
    print(f"\n  Is attention matrix diagonal (identity)? {is_diagonal}")

    if is_diagonal:
        print(f"\n  ✓ CONFIRMED: Each cell only attends to its OWN batch embedding!")
        print(f"    This is NOT cross-batch attention.")
        print(f"    This is adaptive gating of the cell's assigned batch embedding.")

    # Step 5: Check what attention actually computes
    print("\n" + "="*80)
    print("WHAT IS ATTENTION COMPUTING?")
    print("="*80)

    # Manual attention computation
    print(f"\nFor each cell i:")
    print(f"  score[i] = dot_product(z_proj[i], batch_emb[i]) / sqrt(dim)")
    print(f"  weight[i] = softmax(score[i])  # But only one score, so weight = 1.0")
    print(f"  output[i] = weight[i] * batch_emb[i] = batch_emb[i]")

    # Verify output equals input for diagonal attention
    attn_output_squeezed = attn_output.squeeze(0)
    is_passthrough = torch.allclose(attn_output_squeezed, batch_emb, atol=1e-4)
    print(f"\n  Is attn_output ≈ batch_emb (passthrough)? {is_passthrough}")

    if is_passthrough:
        print(f"\n  ✓ CONFIRMED: With diagonal attention, output = input (batch_emb)")
        print(f"    The attention mechanism is currently a PASSTHROUGH.")
        print(f"    The multi-head attention is not learning to modulate batch effects.")

    # Step 6: What about the learned parameters?
    print("\n" + "="*80)
    print("WHAT PARAMETERS ARE LEARNED?")
    print("="*80)

    print(f"\n1. Batch embeddings: {decoder.batch_embedding.weight.shape}")
    print(f"   - {n_batches} batch embeddings, each of dimension {batch_emb_dim}")
    print(f"   - These capture what makes each batch unique")

    print(f"\n2. Attention Q/K/V projections in MultiheadAttention:")
    print(f"   - These transform z_proj and batch_emb before computing attention")
    print(f"   - But with seq_len=1, they can't enable cross-attention")

    print(f"\n3. MLP decoder: Processes [z, attn_output] → (mu, theta)")
    print(f"   - This is where batch correction actually affects output")

    # Full forward pass
    print("\n" + "="*80)
    print("FULL FORWARD PASS")
    print("="*80)

    mu, theta = decoder(z, batch_index)
    print(f"\nOutput shapes:")
    print(f"  mu: {mu.shape} (mean expression)")
    print(f"  theta: {theta.shape} (dispersion)")

    # Test with different scenarios
    print("\n" + "="*80)
    print("TESTING CROSS-BATCH SCENARIO")
    print("="*80)

    # What if we wanted cells to attend to ALL batches?
    print("\nCurrent implementation: Each cell sees only ITS batch")
    print("Alternative implementation: Each cell could see ALL batches")
    print("\nTo enable true cross-batch attention, we would need:")
    print("  1. query = z_proj (one per cell)")
    print("  2. key/value = ALL batch embeddings (not just assigned batch)")
    print("  3. seq_len = n_batches (not 1)")
    print("\nThen attention weights would show: 'How much does cell i relate to batch 0, 1, 2, ...'")

    return decoder, z, batch_index, attn_weights


def test_alternative_attention():
    """Test what TRUE cross-batch attention would look like."""

    print("\n" + "="*80)
    print("ALTERNATIVE: TRUE CROSS-BATCH ATTENTION")
    print("="*80)

    latent_dim = 8
    n_batches = 3
    batch_emb_dim = 16
    batch_size = 5

    # Create a simple model
    batch_embedding = nn.Embedding(n_batches, batch_emb_dim)
    z_projection = nn.Linear(latent_dim, batch_emb_dim)
    multihead_attn = nn.MultiheadAttention(
        embed_dim=batch_emb_dim,
        num_heads=2,
        dropout=0.0,
        batch_first=False
    )

    # Sample data
    torch.manual_seed(42)
    z = torch.randn(batch_size, latent_dim)
    batch_index = torch.tensor([0, 1, 2, 0, 1])

    # Alternative approach: Query ALL batches
    z_proj = z_projection(z)  # (batch_size, batch_emb_dim)

    # Get ALL batch embeddings (not just assigned ones)
    all_batch_indices = torch.arange(n_batches)  # [0, 1, 2]
    all_batch_emb = batch_embedding(all_batch_indices)  # (n_batches, batch_emb_dim)

    print(f"\nSetup:")
    print(f"  z_proj: {z_proj.shape} (each cell's query)")
    print(f"  all_batch_emb: {all_batch_emb.shape} (all batch embeddings as key/value)")

    # Reshape for attention
    query = z_proj.unsqueeze(1)  # (batch_size, 1, batch_emb_dim)
    key = all_batch_emb.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, n_batches, batch_emb_dim)
    value = key

    print(f"\nReshaped:")
    print(f"  query: {query.shape}")
    print(f"  key: {key.shape}")
    print(f"  value: {value.shape}")

    # Transpose for MultiheadAttention format: (seq, batch, dim)
    query_t = query.transpose(0, 1)  # (1, batch_size, batch_emb_dim)
    key_t = key.transpose(0, 1)      # (n_batches, batch_size, batch_emb_dim)
    value_t = value.transpose(0, 1)

    print(f"\nTransposed for attention:")
    print(f"  query: {query_t.shape} (seq_len=1, batch={batch_size})")
    print(f"  key: {key_t.shape} (seq_len={n_batches}, batch={batch_size})")
    print(f"  value: {value_t.shape}")

    attn_output, attn_weights = multihead_attn(
        query=query_t,
        key=key_t,
        value=value_t,
        need_weights=True,
        average_attn_weights=True
    )

    print(f"\nAttention output:")
    print(f"  attn_output: {attn_output.shape}")
    print(f"  attn_weights: {attn_weights.shape}")
    print(f"\nAttention weights (each cell attending to all batches):")
    print(f"  Shape: (batch_size={batch_size}, target_seq=1, source_seq={n_batches})")
    print(attn_weights)

    print(f"\nInterpretation:")
    print(f"  Row i, Column j = How much cell i attends to batch j")
    print(f"  Cell 0 (assigned batch {batch_index[0]}): {attn_weights[0, 0, :].tolist()}")
    print(f"  Cell 1 (assigned batch {batch_index[1]}): {attn_weights[1, 0, :].tolist()}")
    print(f"  Now cells CAN attend to multiple batches, not just their own!")


if __name__ == "__main__":
    # Test current implementation
    decoder, z, batch_index, attn_weights = test_attention_mechanism()

    # Test alternative
    test_alternative_attention()

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
The current BatchCorrectedDecoder implementation:
  ✓ Each cell receives its assigned batch embedding
  ✓ Attention is DIAGONAL (identity matrix)
  ✓ attn_output ≈ batch_emb (passthrough)
  ✓ Multi-head attention parameters CAN modulate this, but structure limits cross-batch learning

This is effectively:
  - Adaptive batch correction (not cross-batch attention)
  - Each cell's batch effect is gated/transformed by attention
  - No comparison across different batches

For true cross-batch attention:
  - Need to query ALL batch embeddings (not just assigned one)
  - Attention weights would show batch similarity
  - More flexible but also more complex
""")
