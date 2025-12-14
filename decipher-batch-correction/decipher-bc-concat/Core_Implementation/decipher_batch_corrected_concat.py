"""
Batch-Corrected Decipher Model with Concatenation

This module extends the original Decipher model to incorporate batch correction
using a simple concatenation approach in the decoder path (z -> x).

This is a simpler baseline compared to the attention-based approach.
"""

import dataclasses
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
import torch.nn as nn
from torch.distributions import constraints
from torch.nn.functional import softplus

import sys
import os

# Import ConditionalDenseNN from local copy instead of decipher package
from conditional_dense_nn import ConditionalDenseNN
from batch_corrected_decoder_concat import BatchCorrectedDecoderConcat


@dataclass(unsafe_hash=True)
class DecipherBatchCorrectedConcatConfig:
    """Configuration for concatenation-based batch-corrected Decipher model."""

    dim_z: int = 10
    dim_v: int = 2

    # Neural network architecture for first decoder (v-->z), neural network f
    layers_v_to_z: Sequence = (64,)

    # Neural network architecture for second decoder (z-->x) is now handled by BatchCorrectedDecoderConcat
    # These parameters control the batch-corrected decoder
    n_batches: int = None  # Number of unique batches (REQUIRED)
    batch_emb_dim: int = 64  # Dimension of batch embeddings
    decoder_hidden_dims: Sequence = (128, 256)  # Hidden layers in decoder MLP
    decoder_dropout: float = 0.1
    use_batch_norm: bool = True  # Use batch normalization in decoder

    beta: float = 1e-1
    seed: int = 0

    learning_rate: float = 5e-3
    val_frac: float = 0.1
    batch_size: int = 64
    n_epochs: int = 1000
    early_stopping_patience: Optional[int] = 10

    dim_genes: int = None
    n_cells: int = None
    prior: str = "normal"

    _initialized_from_adata: bool = False

    def initialize_from_adata(self, adata, batch_key=None):
        """
        Initialize configuration from AnnData object.

        Parameters
        ----------
        adata : AnnData
            Annotated data object
        batch_key : str, optional
            Key in adata.obs containing batch information
            If provided, will automatically set n_batches
        """
        self.dim_genes = adata.shape[1]
        self.n_cells = adata.shape[0]

        if batch_key is not None:
            if batch_key not in adata.obs:
                raise ValueError(f"batch_key '{batch_key}' not found in adata.obs")
            # Get number of unique batches
            self.n_batches = len(adata.obs[batch_key].unique())

        if self.n_batches is None:
            raise ValueError(
                "n_batches must be set. Either provide batch_key in initialize_from_adata "
                "or set n_batches manually in the config."
            )

        self._initialized_from_adata = True

    def to_dict(self):
        res = dataclasses.asdict(self)
        res["layers_v_to_z"] = list(res["layers_v_to_z"])
        res["decoder_hidden_dims"] = list(res["decoder_hidden_dims"])
        return res


class DecipherBatchCorrectedConcat(nn.Module):
    """
    Batch-corrected Decipher model using concatenation for single-cell data.

    This model extends the original Decipher by incorporating batch correction
    in the decoder path using a simple concatenation approach.

    Architecture:
        Encoder (guide):
            x -> z (encoder_x_to_z)
            [z, x] -> v (encoder_zx_to_v)

        Decoder (model):
            v -> z (decoder_v_to_z)
            [z, batch] -> x (decoder_z_to_x with concatenation-based batch correction)

    Key Difference from Attention-Based Approach:
        - Uses simple concatenation: concat([z, batch_embedding]) -> MLP
        - No attention mechanism
        - Simpler architecture with fewer parameters
        - Faster training and inference
        - Good baseline for comparison

    Parameters
    ----------
    config : DecipherBatchCorrectedConcatConfig or dict
        Configuration for the batch-corrected Decipher model.
    """

    def __init__(
        self,
        config: Union[DecipherBatchCorrectedConcatConfig, dict] = DecipherBatchCorrectedConcatConfig(),
    ):
        super().__init__()

        if isinstance(config, dict):
            config = DecipherBatchCorrectedConcatConfig(**config)

        if not config._initialized_from_adata:
            raise ValueError(
                "DecipherBatchCorrectedConcatConfig must be initialized from an AnnData object. "
                "Use `config.initialize_from_adata(adata, batch_key='batch')` to do so."
            )

        self.config = config
        self.dummy_param = nn.Parameter(torch.empty(0))

        # ============== ENCODER (GUIDE) ==============
        # Encoder: x -> z (posterior q(z|x))
        self.encoder_x_to_z = ConditionalDenseNN(
            self.config.dim_genes,
            [128],
            [self.config.dim_z] * 2
        )

        # Encoder: [z, x] -> v (posterior q(v|z,x))
        self.encoder_zx_to_v = ConditionalDenseNN(
            self.config.dim_genes + self.config.dim_z,
            [128],
            [self.config.dim_v, self.config.dim_v],
        )

        # ============== DECODER (MODEL) ==============
        # Decoder: v -> z (prior p(z|v))
        self.decoder_v_to_z = ConditionalDenseNN(
            self.config.dim_v,
            self.config.layers_v_to_z,
            [self.config.dim_z] * 2
        )

        # Decoder: [z, batch] -> x (likelihood p(x|z,batch))
        # THIS IS THE KEY MODIFICATION: Use BatchCorrectedDecoderConcat
        self.decoder_z_to_x = BatchCorrectedDecoderConcat(
            latent_dim=self.config.dim_z,
            n_output=self.config.dim_genes,
            n_batches=self.config.n_batches,
            batch_emb_dim=self.config.batch_emb_dim,
            hidden_dims=list(self.config.decoder_hidden_dims),
            dropout=self.config.decoder_dropout,
            use_batch_norm=self.config.use_batch_norm,
            use_residual=False
        )

        self._epsilon = 1e-5
        self.theta = None

    @property
    def device(self):
        return self.dummy_param.device

    def model(self, x, batch_index, context=None):
        """
        Generative model p(x, z, v | batch).

        Parameters
        ----------
        x : torch.Tensor
            Gene expression data, shape (batch_size, dim_genes)
        batch_index : torch.Tensor
            Batch indices, shape (batch_size,)
        context : torch.Tensor, optional
            Additional context (not used in this version)
        """
        pyro.module("decipher", self)

        with pyro.plate("batch", len(x)), poutine.scale(scale=1.0):
            # Sample v from prior p(v)
            with poutine.scale(scale=self.config.beta):
                if self.config.prior == "normal":
                    prior = dist.Normal(0, x.new_ones(self.config.dim_v)).to_event(1)
                elif self.config.prior == "gamma":
                    prior = dist.Gamma(0.3, x.new_ones(self.config.dim_v) * 0.8).to_event(1)
                else:
                    raise ValueError("Invalid prior, must be normal or gamma")
                v = pyro.sample("v", prior)

            # Sample z from conditional prior p(z|v)
            z_loc, z_scale = self.decoder_v_to_z(v, context=context)
            z_scale = softplus(z_scale)
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

            # Generate x from conditional likelihood p(x|z, batch)
            # KEY CHANGE: Pass batch_index to the decoder
            mu, theta = self.decoder_z_to_x(z, batch_index)

            # Store theta as a pyro parameter (for tracking)
            self.theta = theta.mean(dim=0)

            # Negative Binomial likelihood
            # The BatchCorrectedDecoderConcat outputs mu (mean expression)
            # and theta (dispersion), so we use them directly
            #
            # PyTorch NB parametrization: NegativeBinomial(total_count, logits)
            # where mean = total_count * sigmoid(logits)
            #
            # We have: mean = mu, dispersion = theta
            # Convert to PyTorch parametrization:
            logit = torch.log(mu + self._epsilon) - torch.log(theta + self._epsilon)

            x_dist = dist.NegativeBinomial(
                total_count=theta + self._epsilon,
                logits=logit
            )
            pyro.sample("x", x_dist.to_event(1), obs=x)

    def guide(self, x, batch_index, context=None):
        """
        Inference model q(z, v | x, batch).

        Parameters
        ----------
        x : torch.Tensor
            Gene expression data, shape (batch_size, dim_genes)
        batch_index : torch.Tensor
            Batch indices, shape (batch_size,)
        context : torch.Tensor, optional
            Additional context (not used in this version)

        Returns
        -------
        z_loc : torch.Tensor
            Mean of posterior q(z|x)
        v_loc : torch.Tensor
            Mean of posterior q(v|z,x)
        z_scale : torch.Tensor
            Scale of posterior q(z|x)
        v_scale : torch.Tensor
            Scale of posterior q(v|z,x)
        """
        pyro.module("decipher", self)

        with pyro.plate("batch", len(x)), poutine.scale(scale=1.0):
            # Log-transform input
            x_transformed = torch.log1p(x)

            # Posterior q(z|x)
            z_loc, z_scale = self.encoder_x_to_z(x_transformed, context=context)
            z_scale = softplus(z_scale) + self._epsilon
            posterior_z = dist.Normal(z_loc, z_scale).to_event(1)
            z = pyro.sample("z", posterior_z)

            # Posterior q(v|z,x)
            zx = torch.cat([z, x_transformed], dim=-1)
            v_loc, v_scale = self.encoder_zx_to_v(zx, context=context)
            v_scale = softplus(v_scale) + self._epsilon

            with poutine.scale(scale=self.config.beta):
                if self.config.prior == "gamma":
                    posterior_v = dist.Gamma(softplus(v_loc), v_scale).to_event(1)
                elif self.config.prior == "normal" or self.config.prior == "student-normal":
                    posterior_v = dist.Normal(v_loc, v_scale).to_event(1)
                else:
                    raise ValueError("Invalid prior, must be normal or gamma")
                pyro.sample("v", posterior_v)

        return z_loc, v_loc, z_scale, v_scale

    def compute_v_z_numpy(self, x: np.array):
        """
        Compute decipher_v and decipher_z for a given input.

        Parameters
        ----------
        x : np.ndarray or torch.Tensor
            Input data of shape (n_cells, n_genes).

        Returns
        -------
        v : np.ndarray
            Decipher components v of shape (n_cells, dim_v).
        z : np.ndarray
            Decipher latent z of shape (n_cells, dim_z).
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        x = torch.log1p(x)
        z_loc, _ = self.encoder_x_to_z(x)
        zx = torch.cat([z_loc, x], dim=-1)
        v_loc, _ = self.encoder_zx_to_v(zx)
        return v_loc.detach().numpy(), z_loc.detach().numpy()

    def impute_gene_expression_numpy(self, x, batch_index):
        """
        Impute gene expression given input data and batch information.

        Parameters
        ----------
        x : np.ndarray or torch.Tensor
            Input data of shape (n_cells, n_genes).
        batch_index : np.ndarray or torch.Tensor
            Batch indices of shape (n_cells,).

        Returns
        -------
        imputed : np.ndarray
            Imputed gene expression of shape (n_cells, n_genes).
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if isinstance(batch_index, np.ndarray):
            batch_index = torch.tensor(batch_index, dtype=torch.long)

        # Get latent representation
        z_loc, _, _, _ = self.guide(x, batch_index)

        # Decode with batch correction
        mu, theta = self.decoder_z_to_x(z_loc, batch_index)

        return mu.detach().numpy()


# Example usage and testing
if __name__ == "__main__":
    print("Testing DecipherBatchCorrectedConcat model...")

    # Create mock AnnData-like object
    class MockAnnData:
        def __init__(self, n_cells, n_genes, n_batches):
            self.shape = (n_cells, n_genes)
            self.X = np.random.randn(n_cells, n_genes)
            self.obs = {
                'batch': np.random.randint(0, n_batches, n_cells)
            }

    # Model parameters
    n_cells = 100
    n_genes = 500
    n_batches = 3

    # Create mock data
    mock_adata = MockAnnData(n_cells, n_genes, n_batches)

    # Initialize config
    config = DecipherBatchCorrectedConcatConfig(
        dim_z=10,
        dim_v=2,
        n_batches=n_batches,
        batch_emb_dim=32,
        decoder_hidden_dims=[64, 128],
        decoder_dropout=0.1
    )
    config.initialize_from_adata(mock_adata, batch_key='batch')

    # Create model
    model = DecipherBatchCorrectedConcat(config)

    # Test forward pass
    x = torch.randn(16, n_genes).abs()
    batch_index = torch.randint(0, n_batches, (16,))

    print(f"\nInput shapes:")
    print(f"  x: {x.shape}")
    print(f"  batch_index: {batch_index.shape}")

    # Test guide (encoder)
    z_loc, v_loc, z_scale, v_scale = model.guide(x, batch_index)
    print(f"\nGuide outputs:")
    print(f"  z_loc: {z_loc.shape}")
    print(f"  v_loc: {v_loc.shape}")

    # Test decoder directly
    mu, theta = model.decoder_z_to_x(z_loc, batch_index)
    print(f"\nDecoder outputs:")
    print(f"  mu: {mu.shape}")
    print(f"  theta: {theta.shape}")

    # Test imputation
    imputed = model.impute_gene_expression_numpy(x.numpy(), batch_index.numpy())
    print(f"\nImputed expression shape: {imputed.shape}")

    print(f"\nModel has {sum(p.numel() for p in model.parameters()):,} parameters")

    print("\n" + "="*60)
    print("CONCATENATION-BASED APPROACH")
    print("="*60)
    print("Architecture: concat([z, batch_embedding]) -> MLP -> gene_params")
    print("\nAdvantages:")
    print("  + Simple and interpretable")
    print("  + Fewer parameters")
    print("  + Faster training")
    print("  + Good baseline for comparison")
    print("\nLimitations:")
    print("  - Less flexible in modeling complex batch effects")
    print("  - No learned attention weights for interpretability")
    print("  - Fixed integration of batch information")
    print("="*60)
    print("\n✓ All tests passed!")
