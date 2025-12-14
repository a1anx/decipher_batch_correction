import torch
import torch.nn as nn


class BatchCorrectedDecoderConcat(nn.Module):
    """
    Batch-corrected Decoder with simple concatenation approach.

    This decoder incorporates batch effects by concatenating batch embeddings
    with the latent variable z, then passing through an MLP to decode.
    This is a simpler baseline approach compared to the attention mechanism.

    Architecture:
        1. Embed batch indices into fixed-size vectors
        2. Concatenate [z, batch_embedding]
        3. Pass through MLP decoder to predict gene expression parameters
    """

    def __init__(
        self,
        latent_dim: int,
        n_output: int,
        n_batches: int,
        batch_emb_dim: int = 64,
        hidden_dims: list = None,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        use_residual: bool = False
    ):
        """
        Initialize the concatenation-based batch-corrected decoder.

        Args:
            latent_dim: Dimension of latent variable z
            n_output: Number of output genes
            n_batches: Number of unique batches in the dataset
            batch_emb_dim: Dimension of batch embeddings (default: 64)
            hidden_dims: List of hidden layer dimensions for MLP (default: [128, 256])
            dropout: Dropout probability (default: 0.1)
            use_batch_norm: Whether to use batch normalization (default: True)
            use_residual: Whether to add residual connection (not typically used with concat)
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 256]

        self.latent_dim = latent_dim
        self.n_output = n_output
        self.n_batches = n_batches
        self.batch_emb_dim = batch_emb_dim
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual

        # 1. Batch Embedding Layer
        # Maps batch indices to dense embeddings
        self.batch_embedding = nn.Embedding(n_batches, batch_emb_dim)

        # 2. Build MLP Decoder
        # Input: concatenation of [z, batch_embedding]
        mlp_input_dim = latent_dim + batch_emb_dim

        layers = []
        prev_dim = mlp_input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Output layer: generates parameters for Negative Binomial distribution
        # Output dim = n_output * 2 (for mu and theta)
        layers.append(nn.Linear(prev_dim, n_output * 2))

        self.mlp = nn.Sequential(*layers)

        # Optional residual projection (rarely used with concatenation)
        if use_residual:
            self.residual_proj = nn.Linear(latent_dim, n_output * 2)

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
        Forward pass of the concatenation-based batch-corrected decoder.

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

        # Step 2: Concatenate z and batch embedding
        # This is the key difference from the attention approach:
        # We simply concatenate rather than using attention to learn interactions
        combined = torch.cat([z, batch_emb], dim=-1)  # (batch_size, latent_dim + batch_emb_dim)

        # Step 3: Pass through MLP decoder
        output = self.mlp(combined)  # (batch_size, n_output * 2)

        # Step 4: Optional residual connection
        if self.use_residual:
            residual = self.residual_proj(z)
            output = output + residual

        # Step 5: Split into mu and theta parameters
        mu, theta = torch.split(output, self.n_output, dim=-1)

        # Step 6: Apply activations to ensure valid parameter ranges
        # mu: mean expression (must be positive)
        # theta: dispersion parameter (must be positive)
        mu = torch.exp(mu)
        theta = torch.exp(theta)

        return mu, theta


# Example usage and testing
if __name__ == "__main__":
    # Model parameters
    latent_dim = 32
    n_genes = 2000
    n_batches = 5
    batch_size = 64

    # Create decoder
    decoder = BatchCorrectedDecoderConcat(
        latent_dim=latent_dim,
        n_output=n_genes,
        n_batches=n_batches,
        batch_emb_dim=64,
        hidden_dims=[128, 256],
        dropout=0.1,
        use_batch_norm=True
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

    # Compare parameter count with attention version
    print("\nKey differences from attention-based decoder:")
    print("  1. No multi-head attention mechanism")
    print("  2. Simple concatenation of z and batch embeddings")
    print("  3. Fewer parameters and simpler architecture")
    print("  4. Faster training and inference")
    print("  5. Less flexible in modeling complex batch effects")
