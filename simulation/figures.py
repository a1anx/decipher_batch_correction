import pandas as pd
import scanpy as sc
import os

from matplotlib import pyplot as plt
import seaborn as sns

import matplotlib as mpl

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


def plot_embedding(adata, latent_space, figsize, folder, file_suffix="", title=None):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    adata.obs["branch_id"] = adata.obs["branch_id"].astype(str)
    adata.obs["branch_id"].replace(
        {
            "0": "Origin",
            "1": "Branch 1",
            "-1": "Branch 2",
        },
        inplace=True,
    )
    if latent_space == "decipher_decipher_v":
        import decipher as dc

        dc.tl.decipher_rotate_space(
            adata,
            v1_col="latent_t",
            obsm_key_v="decipher_decipher_v",
            auto_flip_decipher_z=False,
        )

    sc.pl.embedding(
        adata,
        basis=latent_space,
        color="latent_t",
        wspace=0.5,
        hspace=0.5,
        ax=ax,
        size=30,
        show=False,
    )
    # adjust colorbar ticks to just 0 and 1
    cbar = fig.get_axes()[1]
    cbar.set_yticks([0, 1])
    sns.despine()
    # remove x and y axis and their ticks/labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    titles = dict(
        latent="Ground truth",
        X_umap="UMAP",
        X_fd="Force-directed",
        X_scVI_umap="scVI UMAP",
        decipher_decipher_v="Decipher $v$",
    )
    # add margin below title
    if title is None:
        title = titles[latent_space]
    ax.set_title(title, fontsize=12, pad=5)

    save_path = f"{folder}/{latent_space}{file_suffix}.pdf"
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()


folder = "figures/paper"
os.makedirs(folder, exist_ok=True)

# just pick one seed (here the last run)
adata = sc.read("adata/adata_shiftFalse_alpha0.0.h5ad")



latent_spaces = ["latent", "X_umap", "X_scVI_umap", "decipher_decipher_v"]
# X_default_umap, 
# X_pca, 
# X_phate, 
# X_phate_umap, 
# X_scVI, 
# X_scVI_linear, 
# X_scVI_linear_umap, 
# X_scVI_umap, 
# X_umap, 
# decipher_decipher_v, 
# decipher_decipher_z, 
# latent

# figsize = [2.5, 1.5]
# for latent_space in latent_spaces:
#     fig, ax = plt.subplots(1, 1, figsize=figsize)
#     plot_embedding(adata, latent_space, figsize, folder)

# figsize = [2.5, 2]
# poster_folder = "figures/poster"
# os.makedirs(poster_folder, exist_ok=True)
# for latent_space in latent_spaces:
#     fig, ax = plt.subplots(1, 1, figsize=figsize)
#     plot_embedding(adata, latent_space, figsize, poster_folder)

# # Now look at higher densities


if __name__ == "__main__":
    figsize = [2.5, 2]
    out_folder = "figures_delta_shift"
    adata = sc.read(f"adata/adata_shiftFalse_alpha0.0.h5ad")
    sc.pl.embedding(
            adata,
            basis="X_default_umap",
            color=["cluster_latent", "latent_t", "branch_id"],
            ncols=3,
            show=False,
        )
    plt.savefig(
            os.path.join(out_folder, f"umap_0.pdf"),
            bbox_inches="tight",
        )
    plt.close()

        # Decipher visible embedding
    sc.pl.embedding(
            adata,
            basis="decipher_decipher_v",
            color=["cluster_latent", "latent_t", "branch_id"],
            ncols=3,
            show=False,
        )
    plt.savefig(
            os.path.join(out_folder, f"decipher_v_0.pdf"),
            bbox_inches="tight",
        )
    plt.close()
    
    alpha_vec = [0.1, 0.2, 0.5]
    for alpha in alpha_vec:
        adata = sc.read(f"adata/adata_shiftTrue_alpha{alpha}.h5ad")
        # Umap
        sc.pl.embedding(
            adata,
            basis="X_default_umap",
            color=["cluster_latent", "latent_t", "branch_id"],
            ncols=3,
            show=False,
        )
        plt.savefig(
            os.path.join(out_folder, f"umap_{alpha}.pdf"),
            bbox_inches="tight",
        )
        plt.close()

        # Decipher visible embedding
        sc.pl.embedding(
            adata,
            basis="decipher_decipher_v",
            color=["cluster_latent", "latent_t", "branch_id"],
            ncols=3,
            show=False,
        )
        plt.savefig(
            os.path.join(out_folder, f"decipher_v_{alpha}.pdf"),
            bbox_inches="tight",
        )
        plt.close()
