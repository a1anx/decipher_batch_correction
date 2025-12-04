import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import phate
import scanpy as sc
import scvi
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

import decipher as dc

from metrics import compute_plot_metrics
from simulations1130 import simulation_correlated_1, run_methods


if __name__ == "__main__":
    # Example: 3D latent (z1, z3, z5); shift along first two dims
    # Adjust delta to match adata_true.obsm["latent"].shape[1] (usually 3 in your code).
    
    folder = "figures"
    os.makedirs(folder, exist_ok=True)

# v2: n_genes=100, n_samples=2000, sigma=0.05, n_holes=3, hole_size=1
# v3: n_genes=50, n_samples=1000, sigma=0.05, n_holes=3, hole_size=1

    seed = 0
    hole_density = 0.0
    metrics = []

    adata_sim = simulation_correlated_1(
        n_samples=1000,
        n_genes=50,
        seed=seed,
        sigma=0.05,
        branch_prob=0.7,
        k_clusters=20,
        hole_size=1,
        n_holes=3,
        hole_density=hole_density,
    )

    sc.pl.embedding(
        adata_sim,
        basis="latent",
        color=["latent_t", "branch_id", *adata_sim.uns["latent_z_names"]],
        ncols=2,
        show=False,
    )
    plt.savefig(f"{folder}/latent_{hole_density}_seed_{seed}.pdf", bbox_inches="tight")
    #
    latent_spaces = run_methods(adata_sim)
    #
    latent_spaces = [x for x in latent_spaces if adata_sim.obsm[x].shape[1] == 2]
    metrics_local = compute_plot_metrics(
        adata_sim,
        latent_spaces,
        ref_key="latent",
        color=["latent_t", "branch_id"],
    )

    print(metrics_local.info())
# metrics_local = metrics_local.melt(var_name="method", value_name="global_distortion")
# metrics_local["hole_density"] = hole_density
# metrics_local["seed"] = seed
# metrics.append(metrics_local)

# adata_sim.uns["distortion"] = metrics_local
# # save adata
# adata_sim.write(f"{folder}/adata_density_{hole_density}_seed_{seed}.h5ad")
