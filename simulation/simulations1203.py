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

sc.set_figure_params(figsize=[3, 3])

class RandomNet(nn.Module):
    def __init__(self, n_in, n_out, seed=0):
        super().__init__()
        torch.manual_seed(seed)

        self.fc1 = nn.Linear(n_in, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_out)
        self.n_out = n_out

        # initialize weights
        nn.init.normal_(self.fc1.weight, 0, 10)
        nn.init.normal_(self.fc1.bias, 0, 0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc2.bias, 0, 0)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.normal_(self.fc3.bias, 0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        l = torch.normal(np.log(10_000), 0.1, size=(self.n_out,)).exp()
        x = l * x
        return x


def simulation_basic(n_samples=1_000, n_genes=200, seed=0, k_clusters=20) -> sc.AnnData:
    np.random.seed(seed)
    torch.manual_seed(seed)

    # latent time between 0 and 1
    latent_t = np.random.uniform(0, 1, size=(n_samples, 1))

    branching_time = 0.6
    branch = np.random.binomial(1, 0.5, size=(n_samples, 1)) * 2 - 1
    branch = branch * (latent_t > branching_time)
    latent_z = np.random.normal(
        latent_t * np.array([[1, 0]]) + branch * (latent_t - branching_time) * np.array([[0, 1]]),
        0.05,
    )

    net = RandomNet(2, n_genes)
    data = net(torch.tensor(latent_z).float()).detach().numpy().astype(int)

    sim_adata = sc.AnnData(data)
    sim_adata.obs["latent_t"] = latent_t
    sim_adata.obs["branch_id"] = branch
    sim_adata.obs["latent_z1"] = latent_z[:, 0]
    sim_adata.obs["latent_z2"] = latent_z[:, 1]
    sim_adata.obsm["latent"] = latent_z

    sim_adata.obs["cluster_latent"] = (
        KMeans(
            k_clusters,
        )
        .fit(latent_z[:, :2])
        .labels_
    )

    return sim_adata


def simulation_correlated_1(
    n_samples=500,
    n_genes=50,
    seed=0,
    sigma=0.03,
    branch_prob=0.7,
    k_clusters=20,
    hole_size=0,
    n_holes=0,
    hole_density=0.0,
    apply_shift = False,
    alpha: float = 1.0, # slope of the linear shift
) -> sc.AnnData:
    np.random.seed(seed=seed)

    chunk_size = np.array([1, hole_size] * n_holes + [1])
    chunk_offset = np.cumsum(chunk_size) - chunk_size[0]
    chunk_prob = np.array([1, hole_size * hole_density] * n_holes + [1])
    chunk_prob = chunk_prob / chunk_prob.sum()
    total_size = sum(chunk_size)
    # sample which chunk the latent time is in
    chunk = np.random.choice(np.arange(len(chunk_prob)), size=(n_samples,), p=chunk_prob)
    # sample the latent time within the chunk
    latent_t_chunk = np.random.uniform(0, 1, size=(n_samples,))
    latent_t = latent_t_chunk * chunk_size[chunk] + chunk_offset[chunk]
    latent_t = latent_t / total_size
    latent_t = latent_t[:, None]

    # branching
    branching_t1 = 0.3
    branching_t2 = 0.5
    branch_id = np.random.binomial(1, branch_prob, size=(n_samples, 1)) * 2 - 1
    branch_id = branch_id * (latent_t > branching_t1)

    z1 = latent_t * total_size
    z5 = branch_id * (latent_t - branching_t1)
    z3 = (branch_id == 1) * np.clip(latent_t - 0.3, 0, 0.5)
    z4 = (branch_id == 1) * np.clip(latent_t - branching_t2, 0, branching_t1)
    z2 = (branch_id >= 0) * (np.clip(latent_t - 0.2, 0, 0.5) - np.clip(latent_t - 0.5, 0, 0.5))

    # Create latent_z
    latent_z = np.concatenate([z1, z3, z5], axis=1)
    latent_z_sampled = np.random.normal(latent_z, sigma)
    latent_z_sampled[:, 0] /= total_size
    
    
    # delta
    if apply_shift == False:
        delta = 0
    else:
        # latent_t has shape (n_samples, 1)
        delta = alpha * (latent_t[:, 0]) # shape (n_samples,)
    # apply the shift
    latent_z_sampled[:, 0] += delta

    net = RandomNet(
        latent_z_sampled.shape[1],
        n_genes,
    )
    data = net(torch.tensor(latent_z_sampled).float()).detach().numpy().astype(int)

    adata_sim = sc.AnnData(data)
    adata_sim.obs["latent_t"] = latent_t
    adata_sim.obs["branch_id"] = branch_id
    adata_sim.obsm["latent_z"] = latent_z
    
    k_means = KMeans(k_clusters)
    k_means.fit(latent_z)
    adata_sim.obs["cluster_latent"] = k_means.labels_

    for i in range(latent_z.shape[1]):
        adata_sim.obs[f"latent_z{i}"] = latent_z[:, i]
    adata_sim.obsm["latent"] = latent_z_sampled
    latent_names = [f"latent_z{i}" for i in range(latent_z.shape[1])]
    adata_sim.uns["latent_z_names"] = latent_names

    # for each cluster, order the other clusters by distance of their kmeans center
    cluster_centers = k_means.cluster_centers_
    cluster_rank = np.argsort(
        np.linalg.norm(cluster_centers[:, None] - cluster_centers[None, :], axis=2)
    )
    adata_sim.uns["cluster_rank"] = cluster_rank[:, 1:]

    return adata_sim


_LOGGER = logging.getLogger(__name__)


def run_methods(adata, seed=0):
    latent_spaces = []
    
    # ---------- 1) Make a normalized copy for UMAP/PCA/PHATE ----------
    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)
    # optionally: sc.pp.highly_variable_genes, sc.pp.scale, etc.

    # ---------- 2) UMAP on normalized data ----------
    _LOGGER.info("Computing UMAP on normalized data")
    sc.pp.neighbors(adata_norm, random_state=seed)
    sc.tl.umap(adata_norm, random_state=seed)
    adata.obsm["X_norm_umap"] = adata_norm.obsm["X_umap"]
    latent_spaces.append("X_norm_umap")
    _LOGGER.info("Norm UMAP computed")

    # Compute UMAP
    _LOGGER.info("Computing UMAP")
    sc.pp.neighbors(adata, random_state=seed)
    sc.tl.umap(adata, random_state=seed)
    adata.obsm["X_default_umap"] = adata.obsm["X_umap"]
    latent_spaces.append("X_default_umap")
    _LOGGER.info("UMAP computed")

    # Compute scVIe
    # _LOGGER.info("Computing scVI")
    # scvi.settings.seed = seed
    # scvi.model.SCVI.setup_anndata(adata)
    # sim_model = scvi.model.SCVI(adata, gene_likelihood="nb", n_latent=10)
    # sim_model.train()
    # adata.obsm["X_scVI"] = sim_model.get_latent_representation()
    # latent_spaces.append("X_scVI")
    # _LOGGER.info("scVI computed")

    # # Compute scVI-umap
    # _LOGGER.info("Computing scVI-UMAP")
    # sc.pp.neighbors(adata, use_rep="X_scVI", key_added="scVI_neighbors", random_state=seed)
    # sc.tl.umap(adata, neighbors_key="scVI_neighbors", random_state=seed)
    # adata.obsm["X_scVI_umap"] = adata.obsm["X_umap"]
    # latent_spaces.append("X_scVI_umap")
    # _LOGGER.info("scVI-UMAP computed")

    # # Compute scVI-linear
    # _LOGGER.info("Computing scVI-linear")
    # scvi.settings.seed = seed
    # scvi.model.LinearSCVI.setup_anndata(adata)
    # sim_model = scvi.model.LinearSCVI(adata, gene_likelihood="nb", n_latent=10)
    # sim_model.train()
    # adata.obsm["X_scVI_linear"] = sim_model.get_latent_representation()
    # latent_spaces.append("X_scVI_linear")
    # _LOGGER.info("scVI-linear computed")

    # # Compute scVI-linear-umap
    # _LOGGER.info("Computing scVI-linear-UMAP")
    # sc.pp.neighbors(
    #     adata, use_rep="X_scVI_linear", key_added="scVI_linear_neighbors", random_state=seed
    # )
    # sc.tl.umap(adata, neighbors_key="scVI_linear_neighbors", random_state=seed)
    # adata.obsm["X_scVI_linear_umap"] = adata.obsm["X_umap"]
    # latent_spaces.append("X_scVI_linear_umap")
    # _LOGGER.info("scVI-linear-UMAP computed")

    # # Compute PCA
    # _LOGGER.info("Computing PCA")
    # sc.tl.pca(adata, n_comps=10, random_state=seed)
    # latent_spaces.append("X_pca")
    # _LOGGER.info("PCA computed")

    # # Compute Phate
    # _LOGGER.info("Computing Phate")
    # phate_op = phate.PHATE(n_components=10, random_state=seed)
    # adata.obsm["X_phate"] = phate_op.fit_transform(adata.X)
    # latent_spaces.append("X_phate")
    # _LOGGER.info("Phate computed")
    # #
    # _LOGGER.info("Computing Phate-UMAP")
    # sc.pp.neighbors(adata, use_rep="X_phate", key_added="phate_neighbors", random_state=seed)
    # sc.tl.umap(adata, neighbors_key="phate_neighbors", random_state=seed)
    # adata.obsm["X_phate_umap"] = adata.obsm["X_umap"]
    # latent_spaces.append("X_phate_umap")
    # _LOGGER.info("Phate-UMAP computed")

    configuration = {"dim_v": 2, "dim_z": 10}
    config = dc.tl.DecipherConfig(
        **configuration, n_epochs=1000, early_stopping_patience=10, seed=seed
    )
    a = adata.copy()
    res = dc.tl.decipher_train(a, config, plot_every_k_epochs=-2)
    k = "decipher"
    if config.dim_v > 0:
        adata.obsm[k + "_decipher_v"] = a.obsm["decipher_v"]
        latent_spaces.append(k + "_decipher_v")
    adata.obsm[k + "_decipher_z"] = a.obsm["decipher_z"]
    latent_spaces.append(k + "_decipher_z")

    return latent_spaces


def repeated_benchmark(n_repeats=10, seed=0):
    results = []
    seed_init = seed
    for i in range(n_repeats):
        seed = seed_init + i
        adata_sim = simulation_correlated_1(
            n_samples=500,
            n_genes=50,
            seed=seed + i,
            sigma=0.03,
            branch_prob=0.7,
            k_clusters=20,
            hole_size=1,
            n_holes=3,
        )
        latent_spaces = run_methods(adata_sim, seed=seed + i)
        # latent_spaces = [x for x in latent_spaces if adata_sim.obsm[x].shape[1] == 2]

        local_distortion = compute_plot_metrics(
            adata_sim, latent_spaces, ref_key="latent", show_plots=False
        )["Cluster local distortion"]
        local_distortion = local_distortion.melt(var_name="method", value_name="value")
        local_distortion["seed"] = i
        results.append(local_distortion)

    results = pd.concat(results)

    compute_plot_metrics(
        adata_sim, latent_spaces, ref_key="latent", color=["latent_t", "branch_id"], show_plots=True
    )
    return results


# if __name__ == "__main__":
    from metrics import compute_plot_metrics

    folder = "figures"
    os.makedirs(folder, exist_ok=True)

    # v2: n_genes=100, n_samples=2000, sigma=0.05, n_holes=3, hole_size=1
    # v3: n_genes=50, n_samples=1000, sigma=0.05, n_holes=3, hole_size=1

    metrics = []
    for seed in range(5):
        for hole_density in [0.0, 0.025, 0.05, 0.075, 0.1]:
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
    #         metrics_local = compute_plot_metrics(
    #             adata_sim,
    #             latent_spaces,
    #             ref_key="latent",
    #             color=["latent_t", "branch_id"],
    #         )["Global distortion"]
    #         metrics_local = metrics_local.melt(var_name="method", value_name="global_distortion")
    #         metrics_local["hole_density"] = hole_density
    #         metrics_local["seed"] = seed
    #         metrics.append(metrics_local)

    #         adata_sim.uns["distortion"] = metrics_local
    #         # save adata
    #         adata_sim.write(f"{folder}/adata_density_{hole_density}_seed_{seed}.h5ad")

    # metrics = pd.concat(metrics)

    # metrics.to_csv(f"{folder}/metrics-range-of-densities.csv", index=False)


from sklearn.metrics import adjusted_rand_score
from scipy.sparse.csgraph import minimum_spanning_tree


def shift_latent(adata: sc.AnnData, shift) -> sc.AnnData:
    """
    Return a copy of `adata` where the stored latent coordinates are shifted
    by a constant vector `shift`. Expression counts (adata.X) are unchanged.

    - Shifts adata.obsm["latent"] by `shift`
    - Keeps obs["latent_z{i}"] in sync if present
    """
    if "latent" not in adata.obsm_keys():
        raise KeyError("adata.obsm['latent'] is missing; nothing to shift.")

    adata_shifted = adata.copy()

    latent = adata_shifted.obsm["latent"]
# Store noisy latent (latent_z_sampled) in obsm["latent"] — this is the reference embedding used to compute distortion metrics.
    shift = np.asarray(shift, dtype=float)

    if shift.shape[0] != latent.shape[1]:
        raise ValueError(
            f"shift must have length {latent.shape[1]}, got {shift.shape[0]}"
        )

    latent_shifted = latent + shift[None, :]
    adata_shifted.obsm["latent"] = latent_shifted

# keep per-dimension obs fields in sync if they exist
# ["latent_z0", "latent_z1", ...])
# becuase when you shift the latent,v.obsm["latent"] has changed
# but .obs["latent_z0"], latent_z1, ... still show the old values
    for i in range(latent.shape[1]):
        key = f"latent_z{i}"
        if key in adata_shifted.obs:
            adata_shifted.obs[key] = latent_shifted[:, i]

    return adata_shifted


def compute_mst_on_decipher(adata: sc.AnnData, basis_key: str = "decipher_decipher_v"):
    """
    Compute a minimum spanning tree (MST) on Decipher's visible embedding
    and store it in adata.obsp["decipher_mst"].
    """
    if basis_key not in adata.obsm_keys():
        raise KeyError(f"{basis_key} not found in adata.obsm")

    # Build kNN graph in the Decipher visible space
    sc.pp.neighbors(adata, use_rep=basis_key, key_added="decipher_neighbors")
    G = adata.obsp["decipher_neighbors_connectivities"]

    # Minimum spanning tree on the kNN graph
    mst = minimum_spanning_tree(G)
    adata.obsp["decipher_mst"] = mst


def compute_ari_for_latent_spaces(
    adata: sc.AnnData,
    latent_spaces: list[str],
    label_key: str = "cluster_latent",
    random_state: int = 0,
) -> pd.DataFrame:
    """
    For each latent representation in `latent_spaces`, run KMeans clustering
    and compute ARI vs ground-truth labels in `label_key`.
    """
    if label_key not in adata.obs:
        raise KeyError(f"{label_key} not found in adata.obs")

    labels_true = adata.obs[label_key].to_numpy()
    n_clusters = np.unique(labels_true).size

    rows = []
    for rep in latent_spaces:
        X = adata.obsm[rep]
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels_pred = kmeans.fit_predict(X)
        ari = adjusted_rand_score(labels_true, labels_pred)
        rows.append({"method": rep, "ARI": ari})

    return pd.DataFrame(rows)


def run_delta_shift_experiment(
    n_samples: int = 1000,
    n_genes: int = 50,
    sigma: float = 0.05,
    branch_prob: float = 0.7,
    k_clusters: int = 20,
    hole_size: int = 1,
    n_holes: int = 3,
    hole_density: float = 0.05,
    seed: int = 0,
    out_folder: str = "figures_delta_shift",
    adata_folder: str = "newadata",
    apply_shift = False,
    alpha: float = 0.0, # slope of the linear shift
):
    """
    Implements the TA's suggestion:

    - Simulate synthetic data (delta = 0) using simulation_correlated_1
    - Create a delta-shifted version by shifting only the stored latent
    - Run run_methods on each
    - Compute ARI for each method vs ground-truth cluster_latent
    - Plot:
        * true vs shifted latent space
        * UMAP and Decipher visible embeddings
    - Compute an MST on Decipher's visible embedding as a simple trajectory
    """
    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(adata_folder, exist_ok=True)

    # 1. Simulate base (true) dataset: delta = 0
    adata_sim = simulation_correlated_1(
        n_samples=n_samples,
        n_genes=n_genes,
        seed=seed,
        sigma=sigma,
        branch_prob=branch_prob,
        k_clusters=k_clusters,
        hole_size=hole_size,
        n_holes=n_holes,
        hole_density=hole_density,
        apply_shift = apply_shift,
        alpha=alpha,
    )
    sc.pl.embedding(
        adata_sim,
        basis="latent",
        color=["latent_t", "branch_id", *adata_sim.uns["latent_z_names"]],
        ncols=3,
        show=False,
    )
    
    plt.savefig(os.path.join(out_folder, f"latent_shift_{apply_shift}_alpha{alpha}.pdf"), bbox_inches="tight")
    plt.close()

    # 4. Run all methods on the dataset
    latent_spaces = run_methods(adata_sim, seed=seed)

    # (Optional) keep only 2D spaces if you want pure visual spaces
    latent_spaces_2d = [k for k in latent_spaces if adata_sim.obsm[k].shape[1] == 2]

    # 5. Compute ARI for each method vs ground truth clusters
    ari = compute_ari_for_latent_spaces(
        adata_sim,
        latent_spaces_2d,
        label_key="cluster_latent",
        random_state=seed,
    )
    ari["which_dataset"] = f"shift{apply_shift}_alpha{alpha}"
    #ari_all = pd.concat([ari_true, ari_shifted], ignore_index=True)
    ari_folder = "ari"
    os.makedirs(ari_folder, exist_ok=True)
    ari.to_csv(os.path.join(ari_folder, f"ari_by_method_true_vs_shifted{alpha}.csv"), index=False)

    # save adata
    adata_sim.write(f"{adata_folder}/adata_shift{apply_shift}_alpha{alpha}.h5ad")

if __name__ == "__main__":
    # run_delta_shift_experiment(
    #         n_samples=1000,
    #         n_genes=50,
    #         sigma=0.05,
    #         branch_prob=0.7,
    #         k_clusters=20,
    #         hole_size=1,
    #         n_holes=3,
    #         hole_density=0.05,
    #         seed=0,
    #         out_folder="new_figures",
    #         adata_folder= "newadata",
    #         apply_shift = False,
    #         alpha = 0,
    #         )
    alpha_vec = [1,2]
    for alpha in alpha_vec:
        run_delta_shift_experiment(
            n_samples=1000,
            n_genes=50,
            sigma=0.05,
            branch_prob=0.7,
            k_clusters=20,
            hole_size=1,
            n_holes=3,
            hole_density=0.05,
            seed=0,
            out_folder="new_figures",
            adata_folder= "newadata",
            apply_shift = True,
            alpha = alpha,
            )
    
        