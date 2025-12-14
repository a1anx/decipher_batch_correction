import pandas as pd
import scanpy as sc
import os

from matplotlib import pyplot as plt
import seaborn as sns

import matplotlib as mpl

#from figures import plot_embedding

def umap_norm(
    adata,
    out_folder="umap_figures",
    output_name = "umap_norm",
    unique_id = "",
    seed=0,
):
    os.makedirs(out_folder, exist_ok=True)
    adata_temp = adata.copy()
    sc.pp.normalize_total(adata_temp, target_sum=1e4)
    sc.pp.log1p(adata_temp)
    # optionally: sc.pp.highly_variable_genes, sc.pp.scale, etc.

    # ---------- 2) UMAP on normalized data ----------
    sc.pp.neighbors(adata_temp, random_state=seed)
    sc.tl.umap(adata_temp, random_state=seed)
    adata_temp.obsm["X_norm_umap"] = adata_temp.obsm["X_umap"]
    sc.pl.embedding(
        adata_temp,
        basis="X_norm_umap",
        color=["latent_t", "branch_id", "shift"],
        ncols=3,
        show=False,
    )
    plt.savefig(
        os.path.join(out_folder, f"{output_name}_{unique_id}.png"),
        bbox_inches="tight",
    )
    
    

def plot_concat_figure(
    adata,
    out_folder="",
    output_name = "",
    ):
    os.makedirs(out_folder, exist_ok=True)
    # normalized Umap
    sc.pl.embedding(
        adata,
        basis="X_umap",
        color=["latent_t", "branch_id", "shift"],
        ncols=3,
        show=False,
    )
    plt.savefig(
        os.path.join(out_folder, f"umap_{output_name}.png"),
        bbox_inches="tight",
    )
    plt.close()

    #Decipher visible embedding
    sc.pl.embedding(
        adata,
        basis="decipher_decipher_v",
        color=["latent_t", "branch_id", "shift"],
        ncols=3,
        show=False,
    )
    plt.savefig(
        os.path.join(out_folder, f"decipher_v_{output_name}.png"),
        bbox_inches="tight",
    )
    plt.close()
    

if __name__ == "__main__":
    #figsize = [2.5, 2]
    plot_concat_figure(
        adata = sc.read("simulation/titration/adata/adata_combined_alpha_[0, 5.0, 10.0, 15.0].h5ad"),
        out_folder="viz",
        output_name = "alpha_mag_5")
    plot_concat_figure(
        adata = sc.read("simulation/titration/adata/adata_combined_alpha_[0, 1.5, 3.0, 4.5].h5ad"),
        out_folder="viz",
        output_name = "alpha_mag_1.5")
    plot_concat_figure(
        adata = sc.read("simulation/titration/adata/adata_combined_alpha_[0, 2.0, 4.0, 6.0].h5ad"),
        out_folder="viz",
        output_name = "alpha_mag_2")

    

