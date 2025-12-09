import torch
import torch.nn as nn


class BatchCorrectedDecoder(nn.Module):
    """
    Batch-corrected Decoder with Multi-head Attention mechanism.

    This decoder incorporates batch effects using an attention mechanism where
    the latent variable z (cell state) queries batch embeddings to learn
    batch-specific corrections.
    """

    def __init__(
        self,
        latent_dim: int,
        n_output: int,
        n_batches: int,
        batch_emb_dim: int = 64,
        hidden_dims: list = None,
        n_heads: int = 4,
        dropout: float = 0.1,
        combination_mode: str = "concat",
        use_layer_norm: bool = True,
        use_residual: bool = True
    ):
        """
        Initialize the batch-corrected decoder.

        Args:
            latent_dim: Dimension of latent variable z
            n_output: Number of output genes
            n_batches: Number of unique batches in the dataset
            batch_emb_dim: Dimension of batch embeddings (default: 64)
            hidden_dims: List of hidden layer dimensions for MLP (default: [128, 256])
            n_heads: Number of attention heads (default: 4)
            dropout: Dropout probability (default: 0.1)
            combination_mode: How to combine z and attention output
                            - "concat": concatenate [z, attn_output]
                            - "add": add z + attn_output (requires latent_dim == batch_emb_dim)
            use_layer_norm: Whether to apply layer normalization after attention
            use_residual: Whether to add residual connection from z to combined representation
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 256]

        self.latent_dim = latent_dim
        self.n_output = n_output
        self.n_batches = n_batches
        self.batch_emb_dim = batch_emb_dim
        self.combination_mode = combination_mode
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual

        # 1. Batch Embedding Layer
        self.batch_embedding = nn.Embedding(n_batches, batch_emb_dim)

        # 2. Projection layer for z to match attention dimension
        # This is necessary because MultiheadAttention requires query, key, value
        # to have the same embedding dimension
        if latent_dim != batch_emb_dim:
            self.z_projection = nn.Linear(latent_dim, batch_emb_dim)
            self.needs_projection = True
        else:
            self.z_projection = nn.Identity()
            self.needs_projection = False

        # 3. Multi-head Attention
        # Query: z (cell state), Key/Value: batch embeddings
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=batch_emb_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=False  # Use (seq_len, batch, embed_dim) format
        )

        # 4. Optional Layer Normalization after attention
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(batch_emb_dim)

        # 5. Determine input dimension for MLP based on combination mode
        if combination_mode == "concat":
            mlp_input_dim = latent_dim + batch_emb_dim
        elif combination_mode == "add":
            if latent_dim != batch_emb_dim:
                raise ValueError(
                    "For 'add' combination mode, latent_dim must equal batch_emb_dim. "
                    f"Got latent_dim={latent_dim}, batch_emb_dim={batch_emb_dim}"
                )
            mlp_input_dim = latent_dim
        else:
            raise ValueError(
                f"Unknown combination_mode: {combination_mode}. "
                "Choose 'concat' or 'add'."
            )

        # 6. Build MLP Decoder
        layers = []
        prev_dim = mlp_input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer: generates parameters for Negative Binomial distribution
        # Output dim = n_output * 2 (for mu and theta)
        layers.append(nn.Linear(prev_dim, n_output * 2))

        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)

    def forward(self, z, batch_index):
        """
        Forward pass of the batch-corrected decoder.

        Args:
            z: Latent variable, shape (batch_size, latent_dim)
            batch_index: Batch indices, shape (batch_size,) with integer values [0, n_batches)

        Returns:
            mu: Mean parameters for gene expression, shape (batch_size, n_output)
            theta: Dispersion parameters for NB distribution, shape (batch_size, n_output)
        """
        batch_size = z.shape[0]

        # Step 1: Get batch embeddings
        batch_emb = self.batch_embedding(batch_index)  # (batch_size, batch_emb_dim)

        # Step 2: Project z to attention dimension
        z_proj = self.z_projection(z)  # (batch_size, batch_emb_dim)

        # Step 3: Reshape for MultiheadAttention
        # MultiheadAttention expects (seq_len, batch, embed_dim)
        # Here seq_len = 1 (single token per sample)
        query = z_proj.unsqueeze(0)  # (1, batch_size, batch_emb_dim)
        key = batch_emb.unsqueeze(0)  # (1, batch_size, batch_emb_dim)
        value = batch_emb.unsqueeze(0)  # (1, batch_size, batch_emb_dim)

        # Step 4: Apply Cross-Attention
        # z queries the batch embeddings to extract relevant batch-specific information
        attn_output, attn_weights = self.multihead_attn(
            query=query,
            key=key,
            value=value
        )  # attn_output: (1, batch_size, batch_emb_dim)

        # Step 5: Reshape back to (batch_size, batch_emb_dim)
        attn_output = attn_output.squeeze(0)

        # Step 6: Optional layer normalization
        if self.use_layer_norm:
            attn_output = self.layer_norm(attn_output)

        # Step 7: Combine z with attention output (batch effect)
        if self.combination_mode == "concat":
            # Concatenate original z with batch-corrected representation
            combined = torch.cat([z, attn_output], dim=-1)
        elif self.combination_mode == "add":
            # Add batch effect to cell state
            combined = z + attn_output
            if self.use_residual and not self.needs_projection:
                # Residual connection (only when dimensions match)
                combined = combined + z

        # Step 8: Pass through MLP decoder
        output = self.mlp(combined)  # (batch_size, n_output * 2)

        # Step 9: Split into mu and theta parameters
        mu, theta = torch.split(output, self.n_output, dim=-1)

        # Step 10: Apply activations to ensure valid parameter ranges
        # mu: mean expression (must be positive)
        # theta: dispersion parameter (must be positive)
        mu = torch.exp(mu)
        theta = torch.exp(theta)

        return mu, theta

    def get_attention_weights(self, z, batch_index):
        """
        Get attention weights for interpretability.

        Args:
            z: Latent variable, shape (batch_size, latent_dim)
            batch_index: Batch indices, shape (batch_size,)

        Returns:
            attn_weights: Attention weights, shape (batch_size, n_heads, 1, 1)
        """
        batch_emb = self.batch_embedding(batch_index)
        z_proj = self.z_projection(z)

        query = z_proj.unsqueeze(0)
        key = batch_emb.unsqueeze(0)
        value = batch_emb.unsqueeze(0)

        _, attn_weights = self.multihead_attn(
            query=query,
            key=key,
            value=value,
            need_weights=True,
            average_attn_weights=False  # Return per-head attention weights
        )

        return attn_weights


# Example usage and testing
if __name__ == "__main__":
    # Model parameters
    latent_dim = 32
    n_genes = 2000
    n_batches = 5
    batch_size = 64

    # Create decoder
    decoder = BatchCorrectedDecoder(
        latent_dim=latent_dim,
        n_output=n_genes,
        n_batches=n_batches,
        batch_emb_dim=64,
        hidden_dims=[128, 256],
        n_heads=4,
        dropout=0.1,
        combination_mode="concat"
    )

    # Create sample inputs
    z = torch.randn(batch_size, latent_dim)
    batch_index = torch.randint(0, n_batches, (batch_size,))

    # Forward pass
    mu, theta = decoder(z, batch_index)

    print(f"Input z shape: {z.shape}")
    print(f"Batch index shape: {batch_index.shape}")
    print(f"Output mu shape: {mu.shape}")
    print(f"Output theta shape: {theta.shape}")
    print(f"\nModel has {sum(p.numel() for p in decoder.parameters())} parameters")

    # Test attention weights
    attn_weights = decoder.get_attention_weights(z, batch_index)
    print(f"Attention weights shape: {attn_weights.shape}")
